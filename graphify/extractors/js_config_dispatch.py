"""Conservative owner-traced config dispatch for JavaScript / TypeScript.

`const config = getConfig(type); config.modes[mode].submit(p)` calls whichever
function the selected config object holds. Name matching cannot answer that
(`breaker.onSuccess()` shares a name with a config member), so this pass only
resolves a member call whose receiver it can trace, through a small set of
statically understood constructs, to a finite set of *owner objects* identified
by where they are allocated:

* module-level object literals (nested literals included), local `const` aliases,
  imported bindings (one module hop, through barrels);
* lookup / factory functions whose every return expression is enumerable;
* `table[key] || table.Default`, static and dynamic subscripts over closed objects;
* object spread of known objects; top-level `Object.assign(T, known...)` and
  `T.k = value` statements, interpreted in source order.

A call resolves only when the candidate set is complete: every value on the trace
is known, every owner is *closed* (all own keys known, no unmodelled write, never
escaping to code that could mutate it) and every target is a function with a
graph node. It may then resolve to several targets (at most ``MAX_TARGETS``):
each is a function the receiver can hold at runtime. Anything else stays
unresolved. Edges are ``calls`` / INFERRED with ``context: "config_dispatch"``;
the confidence score is inference metadata (one target 0.85, several 0.75), not
a probability.

Branch conditions that provably exclude an alternative are honoured: an owner
selected by ``T[k]`` under key ``K`` is dropped at a call site guarded by
``k !== "K"`` (directly or through a const bound to that comparison), and the
``|| T.Default`` fallback is dropped where a guard proves ``k`` is one of T's keys.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extractors.base import _make_id, _read_text
from graphify.rcfile import effective_suffix

MAX_TARGETS = 4
MAX_OWNERS = 8
MAX_DEPTH = 40
MAX_ATOMS = 64
MAX_FRAMES = 2
CONTEXT = "config_dispatch"
_JS_SUFFIXES = (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts")
_FUNC_TYPES = ("arrow_function", "function_expression", "function", "generator_function")
_FUNC_SCOPES = _FUNC_TYPES + ("function_declaration", "generator_function_declaration",
                              "method_definition")
_OBJECT_PROTOTYPE_KEYS = frozenset({
    "constructor", "hasOwnProperty", "isPrototypeOf", "propertyIsEnumerable",
    "toLocaleString", "toString", "valueOf", "__proto__", "__defineGetter__",
    "__defineSetter__", "__lookupGetter__", "__lookupSetter__",
})
_PRIMITIVES = ("string", "template_string", "number", "true", "false", "null",
               "undefined", "regex", "array")
_WRAPPERS = ("parenthesized_expression", "as_expression", "satisfies_expression",
             "non_null_expression", "type_assertion")


class _Top(Exception):
    """The value is unknown: the call that needs it stays unresolved."""


class _Obj:
    """One abstract object, identified by its allocation site."""

    __slots__ = ("key", "props", "closed", "module", "owner_nid")

    def __init__(self, key, module, owner_nid):
        self.key = key
        self.props: dict[str, frozenset | None] = {}  # None = unknown value
        self.closed = True
        self.module = module
        self.owner_nid = owner_nid  # node id of the module-level binding, if any


# Atoms: ("obj", _Obj, conds) | ("fn", (node_id, writes_this), conds)
#        | ("fnlocal", site, conds) | ("nonfn", type, conds)
# conds: frozenset of ("eq", var, key) / ("notin", var, frozenset(keys)) — the key
# selections under which the atom is the value.

def _cond_ok(conds) -> bool:
    eq: dict = {}
    notin: dict = {}
    for c in conds:
        if c[0] == "eq":
            if c[1] in eq and eq[c[1]] != c[2]:
                return False
            eq[c[1]] = c[2]
        else:
            notin.setdefault(c[1], set()).update(c[2])
    return not any(v in eq and eq[v] in s for v, s in notin.items())


def _with_conds(atoms, extra):
    if not extra:
        return atoms
    out = set()
    for kind, payload, conds in atoms:
        merged = conds | extra
        if _cond_ok(merged):
            out.add((kind, payload, merged))
    return frozenset(out)


def _refuted(conds, facts) -> bool:
    for c in conds:
        for op, var, lit in facts:
            if var != c[1]:
                continue
            if c[0] == "eq" and ((op == "eq" and lit != c[2]) or (op == "neq" and lit == c[2])):
                return True
            if c[0] == "notin" and op == "eq" and lit in c[2]:
                return True
    return False


def _walk(node, skip_functions=False):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        for c in n.children:
            if skip_functions and c.type in _FUNC_SCOPES:
                continue
            stack.append(c)


def _unwrap(node):
    while node is not None and node.type in _WRAPPERS:
        node = next((c for c in node.named_children if c.type not in (
            "type_annotation", "type_identifier", "predefined_type", "generic_type")), None)
    return node


def _writes_this(fn_node) -> bool:
    body = fn_node.child_by_field_name("body")
    for n in _walk(body if body is not None else fn_node, skip_functions=False):
        if n.type in ("assignment_expression", "augmented_assignment_expression", "update_expression"):
            left = n.child_by_field_name("left") or n.child_by_field_name("argument")
            while left is not None and left.type in ("member_expression", "subscript_expression"):
                left = left.child_by_field_name("object")
            if left is not None and left.type == "this":
                return True
    return False


class _Module:
    def __init__(self, path: Path, source: bytes, root):
        self.path = path
        self.source = source
        self.root = root
        self.top: dict[str, object] = {}   # module-level name -> declaration node
        self.env: dict[str, frozenset | None] = {}
        self.state = "new"                 # new | running | done
        self.modelled: set[int] = set()    # start bytes of interpreted statements
        self.imports: dict[str, tuple] = {}
        self.exports: dict[str, tuple] = {}
        self.star_exports: list[str] = []
        self.calls_by_line: dict[int, list] | None = None
        src = source
        for st in root.named_children:
            node = st
            if st.type == "export_statement":
                decl = st.child_by_field_name("declaration")
                spec = st.child_by_field_name("source")
                is_default = any(c.type == "default" for c in st.children)
                clause = next((c for c in st.named_children if c.type == "export_clause"), None)
                if decl is not None:
                    node = decl
                    names = list(self._declared(decl))
                    for name in names:
                        self.exports[name] = ("local", name)
                    if is_default and names:
                        self.exports["default"] = ("local", names[0])
                elif spec is not None:
                    raw = _read_text(spec, src).strip("'\"`")
                    if clause is None:
                        self.star_exports.append(raw)
                    else:
                        for sp in clause.named_children:
                            ids = [c for c in sp.named_children if c.type in ("identifier", "string")]
                            if ids:
                                self.exports[_read_text(ids[-1], src).strip("'\"")] = (
                                    "from", raw, _read_text(ids[0], src).strip("'\""))
                    continue
                else:
                    if clause is not None:
                        for sp in clause.named_children:
                            ids = [c for c in sp.named_children if c.type == "identifier"]
                            if ids:
                                self.exports[_read_text(ids[-1], src)] = ("local", _read_text(ids[0], src))
                    value = st.child_by_field_name("value")
                    if value is not None and value.type == "identifier" and is_default:
                        self.exports["default"] = ("local", _read_text(value, src))
                    continue
            if node.type == "import_statement":
                spec = node.child_by_field_name("source")
                raw = _read_text(spec, src).strip("'\"`") if spec is not None else ""
                for cl in node.named_children:
                    if cl.type != "import_clause":
                        continue
                    for c in cl.named_children:
                        if c.type == "identifier":
                            self.imports[_read_text(c, src)] = (raw, "default")
                        elif c.type == "namespace_import":
                            ident = next((x for x in c.named_children if x.type == "identifier"), None)
                            if ident is not None:
                                self.imports[_read_text(ident, src)] = (raw, "*")
                        elif c.type == "named_imports":
                            for sp in c.named_children:
                                ids = [x for x in sp.named_children if x.type == "identifier"]
                                if ids:
                                    self.imports[_read_text(ids[-1], src)] = (raw, _read_text(ids[0], src))
                continue
            for name in self._declared(node):
                self.top.setdefault(name, node)

    def _declared(self, node):
        if node.type in ("lexical_declaration", "variable_declaration"):
            for d in node.named_children:
                if d.type == "variable_declarator":
                    name = d.child_by_field_name("name")
                    if name is not None and name.type == "identifier":
                        yield _read_text(name, self.source)
        elif node.type in ("function_declaration", "generator_function_declaration", "class_declaration"):
            name = node.child_by_field_name("name")
            if name is not None:
                yield _read_text(name, self.source)

    def is_top(self, node) -> bool:
        """`node` (a declarator or declaration) sits directly at module level."""
        p = node.parent
        if node.type == "variable_declarator":
            p = p.parent if p is not None else None
        if p is not None and p.type == "export_statement":
            p = p.parent
        return p is not None and p.type == "program"

    def calls_at(self, line: int):
        if self.calls_by_line is None:
            self.calls_by_line = {}
            for n in _walk(self.root):
                if n.type == "call_expression":
                    self.calls_by_line.setdefault(n.start_point[0] + 1, []).append(n)
        return self.calls_by_line.get(line, [])


def _is_const(decl_node, mod) -> bool:
    return decl_node.type == "lexical_declaration" and bool(decl_node.children) \
        and decl_node.children[0].type == "const"


class _Resolver:
    def __init__(self, node_ids: set[str], label_index: dict, parse, resolve_module):
        self.node_ids = node_ids
        self.label_index = label_index  # (resolved path, label) -> node id
        self._parse = parse
        self._resolve_module = resolve_module
        self.modules: dict[Path, _Module | None] = {}
        self.objs: dict[tuple, _Obj] = {}
        self.depth = 0
        self.frames: list[dict] = []
        self.visiting: set = set()
        self._lookup_cache: dict = {}
        self._scope_cache: dict = {}
        self._var_cache: dict = {}
        self._assigned_cache: dict = {}
        self.scan_hits: set | None = None
        self._scanned: dict = {}
        self._spec_cache: dict = {}
        self._holds_cache: dict = {}

    # ── modules ──────────────────────────────────────────────────────────
    def module(self, path) -> _Module | None:
        path = Path(path)
        if path not in self.modules:
            self.modules[path] = None
            if effective_suffix(path) in _JS_SUFFIXES:
                parsed = self._parse(path)
                if parsed is not None:
                    self.modules[path] = _Module(path, parsed[0], parsed[1])
        return self.modules[path]

    def resolve_spec(self, mod: _Module, raw: str) -> Path | None:
        if not raw.startswith("."):
            return None
        key = (str(mod.path.parent), raw)
        if key not in self._spec_cache:
            try:
                target = self._resolve_module(raw, mod.path.parent)
            except Exception:
                target = None
            self._spec_cache[key] = Path(target).resolve() if target is not None else None
        return self._spec_cache[key]

    def export_origin(self, mod: _Module, name: str, depth: int = 0):
        """(module, local name) that `mod` exports as `name`, through barrels."""
        if depth > 8:
            raise _Top
        exp = mod.exports.get(name)
        if exp is None:
            found = []
            for raw in mod.star_exports:
                target = self.resolve_spec(mod, raw)
                tmod = self.module(target) if target else None
                if tmod is None:
                    raise _Top  # an unresolvable `export *` could provide the name
                try:
                    found.append(self.export_origin(tmod, name, depth + 1))
                except _Top:
                    continue
            if len(found) != 1:
                raise _Top
            return found[0]
        if exp[0] == "local":
            return mod, exp[1]
        target = self.resolve_spec(mod, exp[1])
        tmod = self.module(target) if target else None
        if tmod is None:
            raise _Top
        return self.export_origin(tmod, exp[2], depth + 1)

    def import_origin(self, mod: _Module, name: str):
        raw, imported = mod.imports[name]
        if imported == "*":
            raise _Top
        target = self.resolve_spec(mod, raw)
        tmod = self.module(target) if target else None
        if tmod is None:
            raise _Top
        return self.export_origin(tmod, imported)

    def run_module(self, mod: _Module) -> None:
        """Interpret the module's top-level statements once, in source order."""
        if mod.state != "new":
            return
        mod.state = "running"
        saved = self.frames
        self.frames = []
        try:
            for st in mod.root.named_children:
                node = st.child_by_field_name("declaration") if st.type == "export_statement" else st
                if node is None:
                    continue
                if node.type in ("lexical_declaration", "variable_declaration"):
                    for d in node.named_children:
                        name = d.child_by_field_name("name") if d.type == "variable_declarator" else None
                        value = d.child_by_field_name("value") if name is not None else None
                        if name is None or name.type != "identifier" or value is None:
                            continue
                        key = _read_text(name, mod.source)
                        inner = _unwrap(value)
                        try:
                            if inner is not None and inner.type in _FUNC_TYPES:
                                mod.env[key] = frozenset({self.fn_atom(mod, key, inner)})
                            else:
                                mod.env[key] = self.eval(value, mod,
                                                         owner_nid=self.label_index.get((mod.path, key)))
                        except _Top:
                            mod.env[key] = None
                elif node.type == "expression_statement" and node.named_children:
                    try:
                        if self._top_level_write(node.named_children[0], mod):
                            mod.modelled.add(node.start_byte)
                    except _Top:
                        pass
        finally:
            self.frames = saved
            mod.state = "done"

    def _single_obj(self, value):
        objs = [a for a in value if a[0] == "obj"]
        if len(value) == 1 and len(objs) == 1 and not objs[0][2]:
            return objs[0][1]
        return None

    def _top_level_write(self, expr, mod) -> bool:
        if expr.type == "assignment_expression":
            left = expr.child_by_field_name("left")
            right = expr.child_by_field_name("right")
            if left is None or right is None or left.type not in ("member_expression", "subscript_expression"):
                return False
            key = self._static_key(left, mod)
            try:
                base = self.eval(left.child_by_field_name("object"), mod)
            except _Top:
                return False
            target = self._single_obj(base)
            if key is None or target is None or key in _OBJECT_PROTOTYPE_KEYS:
                for a in base:
                    if a[0] == "obj":
                        a[1].closed = False
                return True  # handled: the objects it may write are closed off
            try:
                target.props[key] = self.eval(right, mod)
            except _Top:
                target.props[key] = None
            return True
        if expr.type == "call_expression":
            fn = expr.child_by_field_name("function")
            if fn is None or _read_text(fn, mod.source) != "Object.assign":
                return False
            args = [a for a in expr.child_by_field_name("arguments").named_children if a.type != "comment"]
            if not args:
                return False
            try:
                dest = self._single_obj(self.eval(args[0], mod))
            except _Top:
                dest = None
            if dest is None:
                return False  # unknown target: the escape scan decides
            for src in args[1:]:
                try:
                    source = self._single_obj(self.eval(src, mod))
                except _Top:
                    source = None
                if source is None or not source.closed:
                    dest.closed = False
                    return True
                dest.props.update(source.props)
            return True
        return False

    def module_binding(self, mod: _Module, name: str):
        self.run_module(mod)
        decl = mod.top.get(name)
        if decl is None:
            if name in mod.imports:
                omod, oname = self.import_origin(mod, name)
                return self.module_binding(omod, oname)
            raise _Top
        if decl.type in ("function_declaration", "generator_function_declaration"):
            return frozenset({self.fn_atom(mod, name, decl)})
        if decl.type in ("lexical_declaration", "variable_declaration") and name in mod.env:
            value = mod.env[name]
            if value is None or (not _is_const(decl, mod) and self.reassigned(mod.root, name, mod)):
                raise _Top
            return value
        raise _Top  # read before initialisation, class, or unsupported binding

    def fn_atom(self, mod, name, ast, conds=frozenset()):
        nid = self.label_index.get((mod.path, name + "()"))
        if nid is None:
            return ("fnlocal", (str(mod.path), ast.start_byte), conds)
        writes_this = ast.type != "arrow_function" and _writes_this(ast)
        return ("fn", (nid, writes_this), conds)

    # ── lexical lookup ───────────────────────────────────────────────────
    def lookup(self, ident, mod):
        """(kind, declaring node, name) of the nearest declaration, or None."""
        key = (str(mod.path), ident.start_byte, ident.end_byte)
        if key not in self._lookup_cache:
            self._lookup_cache[key] = self._lookup(ident, mod)
        return self._lookup_cache[key]

    def _block_decls(self, node, mod):
        key = (str(mod.path), node.start_byte, node.end_byte, node.type)
        table = self._scope_cache.get(key)
        if table is None:
            table = {}
            for c in node.named_children:
                target = c.child_by_field_name("declaration") if c.type == "export_statement" else c
                if target is None:
                    continue
                if target.type in ("lexical_declaration", "variable_declaration"):
                    for d in target.named_children:
                        if d.type != "variable_declarator":
                            continue
                        nm = d.child_by_field_name("name")
                        if nm is None:
                            continue
                        if nm.type == "identifier":
                            table.setdefault(_read_text(nm, mod.source),
                                             ("const" if _is_const(target, mod) else "let", d))
                        else:
                            for n in self.pattern_names(nm, mod):
                                table.setdefault(n, ("other", d))
                elif target.type in ("function_declaration", "generator_function_declaration",
                                     "class_declaration"):
                    nm = target.child_by_field_name("name")
                    if nm is not None:
                        table.setdefault(_read_text(nm, mod.source),
                                         ("function" if target.type != "class_declaration" else "other", target))
            self._scope_cache[key] = table
        return table

    def _hoisted_vars(self, body, mod):
        key = (str(mod.path), body.start_byte, body.end_byte)
        table = self._var_cache.get(key)
        if table is None:
            table = {}
            for n in _walk(body, skip_functions=True):
                if n.type == "variable_declaration":
                    for d in n.named_children:
                        if d.type != "variable_declarator":
                            continue
                        nm = d.child_by_field_name("name")
                        if nm is None:
                            continue
                        names = [_read_text(nm, mod.source)] if nm.type == "identifier" else self.pattern_names(nm, mod)
                        for name in names:
                            table.setdefault(name, ("let" if nm.type == "identifier" else "other", d))
            self._var_cache[key] = table
        return table

    def _lookup(self, ident, mod):
        name = _read_text(ident, mod.source)
        node = ident.parent
        while node is not None:
            t = node.type
            if t in _FUNC_SCOPES:
                params = node.child_by_field_name("parameters") or node.child_by_field_name("parameter")
                if params is not None and name in self.pattern_names(params, mod):
                    return ("param", node, name)
                if t == "function_expression" or t == "function":
                    own = node.child_by_field_name("name")
                    if own is not None and _read_text(own, mod.source) == name:
                        return ("other", node, name)
                body = node.child_by_field_name("body")
                if body is not None:  # `var` hoists to the function scope
                    hit = self._hoisted_vars(body, mod).get(name)
                    if hit:
                        return (hit[0], hit[1], name)
            elif t in ("statement_block", "program", "switch_case", "switch_default", "class_body"):
                hit = self._block_decls(node, mod).get(name)
                if hit:
                    return (hit[0], hit[1], name)
                if t == "program":
                    return ("import", node, name) if name in mod.imports else None
            elif t in ("for_statement", "for_in_statement"):
                for c in node.named_children:
                    if c.type in ("lexical_declaration", "variable_declaration"):
                        if self._declares(c, name, mod):
                            return ("other", c, name)
                left = node.child_by_field_name("left")
                if left is not None and left.type != "identifier" and name in self.pattern_names(left, mod):
                    return ("other", node, name)
            elif t == "catch_clause":
                p = node.child_by_field_name("parameter")
                if p is not None and name in self.pattern_names(p, mod):
                    return ("other", node, name)
            node = node.parent
        return None

    def _declares(self, decl, name, mod):
        for d in decl.named_children:
            if d.type != "variable_declarator":
                continue
            nm = d.child_by_field_name("name")
            if nm is None:
                continue
            if nm.type == "identifier":
                if _read_text(nm, mod.source) == name:
                    return ("const" if _is_const(decl, mod) else "let", d, name)
            elif name in self.pattern_names(nm, mod):
                return ("other", d, name)
        return None

    def pattern_names(self, node, mod):
        out = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type in ("identifier", "shorthand_property_identifier_pattern"):
                out.add(_read_text(n, mod.source))
                continue
            if n.type == "pair_pattern":
                v = n.child_by_field_name("value")
                if v is not None:
                    stack.append(v)
                continue
            if n.type in ("assignment_pattern", "object_assignment_pattern"):
                left = n.child_by_field_name("left")
                if left is not None:
                    stack.append(left)
                continue
            if n.type in ("required_parameter", "optional_parameter"):
                pat = n.child_by_field_name("pattern")
                if pat is not None:
                    stack.append(pat)
                continue
            if n.type in ("type_annotation", "property_identifier"):
                continue
            stack.extend(n.named_children)
        return out

    def reassigned(self, scope, name, mod) -> bool:
        key = (str(mod.path), scope.start_byte, scope.end_byte)
        names = self._assigned_cache.get(key)
        if names is None:
            names = set()
            for n in _walk(scope):
                if n.type in ("assignment_expression", "augmented_assignment_expression"):
                    left = n.child_by_field_name("left")
                elif n.type == "update_expression":
                    left = n.child_by_field_name("argument")
                else:
                    continue
                if left is not None and left.type == "identifier":
                    names.add(_read_text(left, mod.source))
            self._assigned_cache[key] = names
        return name in names

    def _scope_of(self, binding):
        kind, node, _ = binding
        p = node if kind == "param" else node.parent
        while p is not None and p.type not in _FUNC_SCOPES + ("program",):
            p = p.parent
        return p

    def var_id(self, ident, mod):
        """Identity of a never-reassigned binding, for key-selection conditions."""
        if ident is None or ident.type != "identifier":
            return None
        b = self.lookup(ident, mod)
        if b is None or b[0] not in ("const", "param", "let"):
            return None
        if b[0] == "param":
            for frame in reversed(self.frames):
                hit = frame.get((str(mod.path), b[1].start_byte, b[2]))
                if hit is not None:
                    return hit[0]
        scope = self._scope_of(b)
        if b[0] != "const" and (scope is None or self.reassigned(scope, b[2], mod)):
            return None
        return (str(mod.path), b[1].start_byte, b[2])

    # ── evaluation ───────────────────────────────────────────────────────
    def eval(self, node, mod, owner_nid=None):
        self.depth += 1
        try:
            if self.depth > MAX_DEPTH or node is None:
                raise _Top
            value = self._eval(node, mod, owner_nid)
            if len(value) > MAX_ATOMS:
                raise _Top
            return value
        finally:
            self.depth -= 1

    def _eval(self, node, mod, owner_nid):
        t = node.type
        if t in _WRAPPERS:
            return self.eval(_unwrap(node), mod, owner_nid)
        if t in _PRIMITIVES:
            return frozenset({("nonfn", t, frozenset())})
        if t == "identifier":
            return self._eval_identifier(node, mod)
        if t == "object":
            return frozenset({("obj", self._object(node, mod, owner_nid), frozenset())})
        if t in ("member_expression", "subscript_expression"):
            return self._eval_member(node, mod)
        if t == "binary_expression":
            op = node.child_by_field_name("operator")
            op = _read_text(op, mod.source) if op is not None else ""
            if op not in ("||", "??"):
                raise _Top
            left_node = node.child_by_field_name("left")
            left = self.eval(left_node, mod)
            right = self.eval(node.child_by_field_name("right"), mod)
            fallback = self._fallback_cond(left_node, mod)
            if fallback is not None:
                right = _with_conds(right, frozenset({fallback}))
            return left | right
        if t == "ternary_expression":
            return (self.eval(node.child_by_field_name("consequence"), mod)
                    | self.eval(node.child_by_field_name("alternative"), mod))
        if t == "call_expression":
            return self._eval_call(node, mod)
        if t in _FUNC_TYPES:
            return frozenset({("fnlocal", (str(mod.path), node.start_byte), frozenset())})
        raise _Top

    def _eval_identifier(self, node, mod):
        name = _read_text(node, mod.source)
        key = (str(mod.path), node.start_byte, tuple(f["_id"] for f in self.frames))
        if key in self.visiting:
            raise _Top
        self.visiting.add(key)
        try:
            b = self.lookup(node, mod)
            if b is None:
                raise _Top
            kind, decl, _ = b
            if kind == "import" or (kind in ("const", "let", "function") and mod.is_top(decl)):
                return self.module_binding(mod, name)
            if kind == "const":
                value = decl.child_by_field_name("value")
                inner = _unwrap(value)
                if inner is not None and inner.type in _FUNC_TYPES:
                    return frozenset({("fnlocal", (str(mod.path), inner.start_byte), frozenset())})
                return self.eval(value, mod)
            if kind == "let":
                scope = self._scope_of(b)
                if scope is None or self.reassigned(scope, name, mod):
                    raise _Top
                return self.eval(decl.child_by_field_name("value"), mod)
            if kind == "function":
                return frozenset({("fnlocal", (str(mod.path), decl.start_byte), frozenset())})
            raise _Top  # parameters, catch bindings, destructured names
        finally:
            self.visiting.discard(key)

    def _static_key(self, member, mod):
        if member.type == "member_expression":
            prop = member.child_by_field_name("property")
            if prop is not None and prop.type == "property_identifier":
                return _read_text(prop, mod.source)
            return None
        index = _unwrap(member.child_by_field_name("index"))
        if index is not None and index.type == "string":
            return _read_text(index, mod.source)[1:-1]
        return None

    def _eval_member(self, node, mod):
        base = self.eval(node.child_by_field_name("object"), mod)
        key = self._static_key(node, mod)
        dyn_var = None
        if key is None:
            index = _unwrap(node.child_by_field_name("index")) if node.type == "subscript_expression" else None
            if index is None or index.type != "identifier":
                raise _Top
            dyn_var = self.var_id(index, mod)
        elif key in _OBJECT_PROTOTYPE_KEYS:
            raise _Top
        out = set()
        for kind, payload, conds in base:
            if kind != "obj" or not payload.closed:
                raise _Top  # property of a function, primitive, array or open object
            obj = payload
            keys = [key] if key is not None else sorted(obj.props)
            for k in keys:
                if k not in obj.props:
                    continue  # absent on a closed object: undefined at runtime
                value = obj.props[k]
                if value is None:
                    raise _Top
                extra = conds | ({("eq", dyn_var, k)} if key is None and dyn_var is not None else set())
                out |= _with_conds(value, frozenset(extra))
        return frozenset(out)

    def _fallback_cond(self, left_node, mod):
        """`T[k] || D`: D is reached only when `k` is not one of T's own keys."""
        left_node = _unwrap(left_node)
        if left_node is None or left_node.type != "subscript_expression" \
                or self._static_key(left_node, mod) is not None:
            return None
        index = _unwrap(left_node.child_by_field_name("index"))
        var = self.var_id(index, mod) if index is not None else None
        if var is None:
            return None
        try:
            base = self.eval(left_node.child_by_field_name("object"), mod)
        except _Top:
            return None
        keys = set()
        for kind, obj, conds in base:
            if kind != "obj" or conds or not obj.closed:
                return None
            for k, v in obj.props.items():
                if v is None or any(a[0] == "nonfn" for a in v):
                    return None  # a falsy own value would also fall through
                keys.add(k)
        return ("notin", var, frozenset(keys))

    def _object(self, node, mod, owner_nid):
        key = (str(mod.path), node.start_byte, tuple(f["_id"] for f in self.frames))
        if key in self.objs:
            return self.objs[key]
        obj = _Obj(key, mod, owner_nid if not self.frames else None)
        self.objs[key] = obj
        for child in node.named_children:
            t = child.type
            if t == "comment":
                continue
            if t == "pair":
                k = child.child_by_field_name("key")
                v = child.child_by_field_name("value")
                if k is None or v is None:
                    obj.closed = False
                    continue
                if k.type in ("property_identifier", "identifier"):
                    name, ident_key = _read_text(k, mod.source), True
                elif k.type == "string":
                    name, ident_key = _read_text(k, mod.source)[1:-1], False
                elif k.type == "computed_property_name" and k.named_children \
                        and k.named_children[0].type == "string":
                    name, ident_key = _read_text(k.named_children[0], mod.source)[1:-1], False
                else:
                    obj.closed = False
                    continue
                inner = _unwrap(v)
                if inner is not None and inner.type in _FUNC_TYPES:
                    obj.props[name] = frozenset({self._member_fn(obj, name, inner, ident_key)})
                elif inner is not None and inner.type == "object":
                    obj.props[name] = frozenset({("obj", self._object(inner, mod, obj.owner_nid), frozenset())})
                else:
                    try:
                        obj.props[name] = self.eval(v, mod)
                    except _Top:
                        obj.props[name] = None
            elif t == "method_definition":
                if any(c.type in ("get", "set") for c in child.children):
                    obj.closed = False
                    continue
                k = child.child_by_field_name("name")
                if k is None or k.type != "property_identifier":
                    obj.closed = False
                    continue
                name = _read_text(k, mod.source)
                obj.props[name] = frozenset({self._member_fn(obj, name, child, True)})
            elif t == "shorthand_property_identifier":
                try:
                    obj.props[_read_text(child, mod.source)] = self.eval(child, mod)
                except _Top:
                    obj.props[_read_text(child, mod.source)] = None
            elif t == "spread_element":
                inner = child.named_children[0] if child.named_children else None
                try:
                    source = self._single_obj(self.eval(inner, mod))
                except _Top:
                    source = None
                if source is None or not source.closed:
                    obj.closed = False
                    continue
                obj.props.update(source.props)
            else:
                obj.closed = False
        return obj

    def _member_fn(self, obj, name, fn_node, ident_key):
        site = (str(obj.module.path), fn_node.start_byte)
        if obj.owner_nid is None or not ident_key:
            return ("fnlocal", site, frozenset())
        nid = _make_id(obj.owner_nid, name)
        if nid not in self.node_ids:
            return ("fnlocal", site, frozenset())
        writes_this = fn_node.type != "arrow_function" and _writes_this(fn_node)
        return ("fn", (nid, writes_this), frozenset())

    def _eval_call(self, node, mod):
        fn = node.child_by_field_name("function")
        if fn is None or fn.type != "identifier" or len(self.frames) >= MAX_FRAMES:
            raise _Top
        b = self.lookup(fn, mod)
        if b is None:
            raise _Top
        name = _read_text(fn, mod.source)
        if b[0] == "import":
            fmod, fname = self.import_origin(mod, name)
        elif b[0] in ("const", "function") and mod.is_top(b[1]):
            fmod, fname = mod, name
        else:
            raise _Top
        self.run_module(fmod)
        fdecl = fmod.top.get(fname)
        if fdecl is None:
            raise _Top
        if fdecl.type == "lexical_declaration" and _is_const(fdecl, fmod):
            d = next((d for d in fdecl.named_children if d.type == "variable_declarator"
                      and d.child_by_field_name("name") is not None
                      and _read_text(d.child_by_field_name("name"), fmod.source) == fname), None)
            func = _unwrap(d.child_by_field_name("value")) if d is not None else None
        elif fdecl.type == "function_declaration":
            func = fdecl
        else:
            raise _Top
        if func is None or func.type not in ("arrow_function", "function_expression", "function",
                                              "function_declaration"):
            raise _Top
        fn_key = (str(fmod.path), func.start_byte)
        if any(f["_fn"] == fn_key for f in self.frames):
            raise _Top  # recursion
        params = func.child_by_field_name("parameters") or func.child_by_field_name("parameter")
        args_node = node.child_by_field_name("arguments")
        args = [a for a in (args_node.named_children if args_node is not None else []) if a.type != "comment"]
        frame = {"_id": (str(mod.path), node.start_byte), "_fn": fn_key}
        plist = [] if params is None else ([params] if params.type == "identifier"
                                           else [p for p in params.named_children if p.type != "comment"])
        arg_vars = [self.var_id(a, mod) if a.type == "identifier" else None for a in args]
        for i, p in enumerate(plist):
            ident = p
            if p.type in ("required_parameter", "optional_parameter"):
                ident = p.child_by_field_name("pattern")
            elif p.type == "assignment_pattern":
                ident = p.child_by_field_name("left")
            if ident is None or ident.type != "identifier":
                continue
            frame[(str(fmod.path), func.start_byte, _read_text(ident, fmod.source))] = (
                arg_vars[i] if i < len(arg_vars) else None,)
        self.frames.append(frame)
        try:
            body = func.child_by_field_name("body")
            if body is None:
                raise _Top
            if body.type != "statement_block":
                return self.eval(body, fmod)
            returns = [n for n in _walk(body, skip_functions=True) if n.type == "return_statement"]
            if not returns:
                raise _Top
            out = set()
            for r in returns:
                arg = next((c for c in r.named_children if c.type != "comment"), None)
                if arg is None:
                    raise _Top
                out |= self.eval(arg, fmod)
            return frozenset(out)
        finally:
            self.frames.pop()


# ── escape analysis ─────────────────────────────────────────────────────

def _value_objs(res, node, mod):
    try:
        value = res.eval(node, mod)
    except _Top:
        return []
    except RecursionError:
        return []
    return [a[1] for a in value if a[0] == "obj"]


def _shared_keys(res, mod) -> set:
    """Objects reachable from the module's top-level bindings: shared state. An
    object a function allocates per call is not, and escaping it affects no one."""
    cache = getattr(res, "_shared_cache", None)
    if cache is None:
        cache = res._shared_cache = {}
    if mod.path not in cache:
        res.run_module(mod)
        seen: set = set()
        stack = [a[1] for v in mod.env.values() for a in v or () if a[0] == "obj"]
        while stack:
            o = stack.pop()
            if o.key in seen:
                continue
            seen.add(o.key)
            for v in o.props.values():
                stack.extend(a[1] for a in v or () if a[0] == "obj")
        cache[mod.path] = seen
    return cache[mod.path]


def _escape(res, obj, deep):
    if res.scan_hits is not None:
        stack, seen = [obj], set()
        while stack:
            o = stack.pop()
            if o.key in seen:
                continue
            seen.add(o.key)
            if o.key in _shared_keys(res, o.module):
                res.scan_hits.add(o.module.path)
            if deep:
                for v in o.props.values():
                    stack.extend(a[1] for a in v or () if a[0] == "obj")
    _close_off(obj, deep)


def _close_off(obj, deep, seen=None):
    seen = set() if seen is None else seen
    if obj.key in seen:
        return
    seen.add(obj.key)
    obj.closed = False
    if deep:
        for value in obj.props.values():
            for a in value or ():
                if a[0] == "obj":
                    _close_off(a[1], True, seen)


def _escape_scan(res: _Resolver, mod: _Module) -> None:
    """Close off every object that a write or an unknown use could change."""
    res.run_module(mod)
    for node in _walk(mod.root):
        t = node.type
        if t in ("assignment_expression", "augmented_assignment_expression", "update_expression") \
                or (t == "unary_expression" and node.children and _read_text(node.children[0], mod.source) == "delete"):
            stmt = node.parent
            if stmt is not None and stmt.type == "expression_statement" and stmt.start_byte in mod.modelled:
                continue
            left = node.child_by_field_name("left") or node.child_by_field_name("argument")
            left = _unwrap(left)
            if left is not None and left.type in ("member_expression", "subscript_expression"):
                for obj in _value_objs(res, left.child_by_field_name("object"), mod):
                    _escape(res, obj, False)
        elif t == "identifier" and _is_value_reference(node):
            _climb(res, node, mod)
        elif t == "shorthand_property_identifier":
            _climb(res, node, mod)
        elif t == "call_expression":
            fn = node.child_by_field_name("function")
            if fn is not None and fn.type == "identifier":
                _climb(res, node, mod)


def _is_value_reference(node) -> bool:
    p = node.parent
    if p is None:
        return False
    if p.type == "variable_declarator" and p.child_by_field_name("name") == node:
        return False
    if p.type in ("import_specifier", "import_clause", "namespace_import", "export_specifier",
                  "formal_parameters", "required_parameter", "optional_parameter",
                  "function_declaration", "function_expression", "class_declaration",
                  "method_definition", "pair_pattern", "object_pattern", "array_pattern",
                  "assignment_pattern", "object_assignment_pattern", "rest_pattern",
                  "labeled_statement", "break_statement", "continue_statement",
                  "jsx_opening_element", "jsx_closing_element", "jsx_self_closing_element",
                  "jsx_attribute", "nested_identifier", "catch_clause"):
        return False
    if p.type == "arrow_function" and p.child_by_field_name("parameter") == node:
        return False
    if p.type == "call_expression" and p.child_by_field_name("function") == node:
        return False
    if p.type in ("assignment_expression", "augmented_assignment_expression") \
            and p.child_by_field_name("left") == node:
        return False
    return True


def _returns_to_scanned_callers(node, mod, objs) -> bool:
    """A value returned from a module-level function of the objects' own module
    reaches only that function's callers, whose use of the result is scanned."""
    p = node.parent
    while p is not None and p.type not in _FUNC_SCOPES:
        p = p.parent
    if p is None:
        return False
    decl = p if p.type == "function_declaration" else p.parent
    if decl is None or decl.type not in ("function_declaration", "variable_declarator"):
        return False
    return mod.is_top(decl) and all(o.module is mod for o in objs)


def _climb(res, node, mod):
    objs = _value_objs(res, node, mod)
    if not objs:
        return
    child, p = node, node.parent
    while p is not None:
        t = p.type
        if t in _WRAPPERS:
            child, p = p, p.parent
            continue
        if t in ("member_expression", "subscript_expression"):
            if p.child_by_field_name("object") != child:
                return  # used as a subscript key
            gp = p.parent
            if gp is not None and gp.type == "call_expression" and gp.child_by_field_name("function") == p:
                return  # the dispatch itself (`this` writes are checked per target)
            nxt = _value_objs(res, p, mod)
            if not nxt:
                return  # read of a function or primitive
            objs, child, p = nxt, p, p.parent
            continue
        if t == "binary_expression":
            op = p.child_by_field_name("operator")
            if op is not None and _read_text(op, mod.source) in ("||", "??", "&&"):
                child, p = p, p.parent
                continue
            return  # comparison: only a test
        if t == "ternary_expression":
            if p.child_by_field_name("condition") == child:
                return
            child, p = p, p.parent
            continue
        if t == "unary_expression":
            return  # `!x`, `typeof x`, `void x`
        if t in ("if_statement", "while_statement", "do_statement", "switch_statement",
                 "expression_statement", "parenthesized_expression"):
            return
        if t == "variable_declarator" and p.child_by_field_name("value") == child:
            decl = p.parent
            name = p.child_by_field_name("name")
            if decl is not None and _is_const(decl, mod) and name is not None and name.type == "identifier":
                return  # a const alias: its own references are scanned
            break
        if t == "spread_element" and p.parent is not None and p.parent.type == "object":
            return  # copied, not shared
        if t in ("pair", "object") and _in_modelled_literal(p, mod):
            return
        if t in ("return_statement", "arrow_function"):
            if _returns_to_scanned_callers(child, mod, objs):
                return
            break
        if t == "arguments" and p.parent is not None:
            call = p.parent
            fn = call.child_by_field_name("function")
            if fn is not None and _read_text(fn, mod.source) == "Object.assign":
                stmt = call.parent
                first = next((a for a in p.named_children if a.type != "comment"), None)
                modelled = stmt is not None and stmt.type == "expression_statement" \
                    and stmt.start_byte in mod.modelled
                if modelled:
                    return  # a modelled source or target
                if first != child:
                    return  # a source is only read
            break
        if t in ("export_statement", "export_clause", "export_specifier"):
            return
        if t == "assignment_expression":
            stmt = p.parent
            if stmt is not None and stmt.type == "expression_statement" and stmt.start_byte in mod.modelled:
                return
            break
        break
    for obj in objs:
        _escape(res, obj, True)


def _scan_file(res, path):
    """Escape-scan one module once; return the other modules whose objects it
    writes or hands to unknown code."""
    path = Path(path)
    if path not in res._scanned:
        mod = res.module(path)
        hits: set = set()
        if mod is not None:
            res.scan_hits = hits
            try:
                _escape_scan(res, mod)
            except Exception:
                for obj in list(res.objs.values()):
                    _close_off(obj, False)  # fail closed: nothing resolves
            finally:
                res.scan_hits = None
        res._scanned[path] = frozenset(h for h in hits if h != path)
    return res._scanned[path]


def _module_holds_functions(mod) -> bool:
    for st in mod.root.named_children:
        decl = st.child_by_field_name("declaration") if st.type == "export_statement" else st
        if decl is None or decl.type not in ("lexical_declaration", "variable_declaration"):
            continue
        for d in decl.named_children:
            value = _unwrap(d.child_by_field_name("value")) if d.type == "variable_declarator" else None
            if value is None or value.type != "object":
                continue
            for n in _walk(value):
                if n.type in _FUNC_TYPES + ("method_definition", "shorthand_property_identifier",
                                            "spread_element"):
                    return True
                if n.type == "pair":
                    v = _unwrap(n.child_by_field_name("value"))
                    if v is not None and v.type in ("identifier", "member_expression", "call_expression"):
                        return True
    return False


def _imports_function_holders(res, mod) -> bool:
    for raw in {raw for raw, _ in mod.imports.values()}:
        target = res.resolve_spec(mod, raw)
        if target is None:
            continue
        if target not in res._holds_cache:
            tmod = res.module(target)
            res._holds_cache[target] = tmod is not None and _module_holds_functions(tmod)
        if res._holds_cache[target]:
            return True
    return False


def _in_modelled_literal(node, mod):
    """A literal nested in a module-level declaration's initialiser is modelled."""
    p = node
    while p is not None:
        if p.type in _FUNC_SCOPES:
            return False
        if p.type == "variable_declarator":
            return mod.is_top(p)
        p = p.parent
    return False


# ── branch-condition facts ──────────────────────────────────────────────

def _site_facts(res, call, mod):
    facts = set()
    child, p = call, call.parent
    while p is not None:
        if p.type == "if_statement":
            cond = p.child_by_field_name("condition")
            if p.child_by_field_name("consequence") == child:
                facts |= _facts(res, cond, mod, True, 0)
            elif p.child_by_field_name("alternative") == child:
                facts |= _facts(res, cond, mod, False, 0)
        elif p.type == "else_clause" and p.parent is not None and p.parent.type == "if_statement":
            facts |= _facts(res, p.parent.child_by_field_name("condition"), mod, False, 0)
        elif p.type == "binary_expression" and p.child_by_field_name("right") == child:
            op = p.child_by_field_name("operator")
            if op is not None and _read_text(op, mod.source) == "&&":
                facts |= _facts(res, p.child_by_field_name("left"), mod, True, 0)
        elif p.type == "ternary_expression":
            cond = p.child_by_field_name("condition")
            if p.child_by_field_name("consequence") == child:
                facts |= _facts(res, cond, mod, True, 0)
            elif p.child_by_field_name("alternative") == child:
                facts |= _facts(res, cond, mod, False, 0)
        if p.type in ("statement_block", "program"):
            for st in p.named_children:
                if st.start_byte >= child.start_byte:
                    break
                if st.type == "if_statement" and st.child_by_field_name("alternative") is None \
                        and _always_exits(st.child_by_field_name("consequence")):
                    facts |= _facts(res, st.child_by_field_name("condition"), mod, False, 0)
        child, p = p, p.parent
    return facts


def _always_exits(node):
    if node is None:
        return False
    if node.type in ("return_statement", "throw_statement"):
        return True
    if node.type == "statement_block":
        return any(c.type in ("return_statement", "throw_statement") for c in node.named_children)
    return False


def _facts(res, node, mod, positive, depth):
    node = _unwrap(node)
    if node is None or depth > 6:
        return set()
    t = node.type
    if t == "unary_expression" and node.children and node.children[0].type == "!":
        return _facts(res, node.child_by_field_name("argument"), mod, not positive, depth + 1)
    if t == "binary_expression":
        op = _read_text(node.child_by_field_name("operator"), mod.source)
        left = _unwrap(node.child_by_field_name("left"))
        right = _unwrap(node.child_by_field_name("right"))
        if op == "&&" and positive:
            return _facts(res, left, mod, True, depth + 1) | _facts(res, right, mod, True, depth + 1)
        if op == "||" and not positive:
            return _facts(res, left, mod, False, depth + 1) | _facts(res, right, mod, False, depth + 1)
        if op in ("===", "==", "!==", "!=") and left is not None and right is not None:
            ident, lit = (left, right) if right.type == "string" else (right, left)
            if lit.type != "string" or ident.type != "identifier":
                return set()
            var = res.var_id(ident, mod)
            if var is None:
                return set()
            eq = (op in ("===", "==")) == positive
            return {("eq" if eq else "neq", var, _read_text(lit, mod.source)[1:-1])}
        return set()
    if t == "identifier":
        b = res.lookup(node, mod)
        if b is not None and b[0] == "const":
            return _facts(res, b[1].child_by_field_name("value"), mod, positive, depth + 1)
    return set()


# ── entry point ─────────────────────────────────────────────────────────

def resolve_config_dispatch(paths, nodes, edges, root, raw_calls, resolution_context_nodes,
                            importers_of, parse, resolve_module, source_path) -> None:
    """Add INFERRED `calls` edges for member calls whose receiver traces to
    config objects. ``importers_of(path)`` returns the modules that import
    ``path`` (fresh facts plus persisted context); they are escape-scanned."""
    res = None
    candidates = []
    for call in raw_calls or ():
        if (not call.get("is_member_call") or call.get("_js_jsx") or call.get("_js_construct")
                or call.get("ts_this_field") or call.get("ts_constructed_type")):
            continue
        receiver = str(call.get("receiver") or "")
        if not receiver.isidentifier() or receiver in ("this", "super") or not call.get("callee"):
            continue
        if effective_suffix(str(call.get("source_file", ""))) not in _JS_SUFFIXES:
            continue
        candidates.append(call)

    all_nodes = list(nodes) + list(resolution_context_nodes or ())
    node_ids = {str(n.get("id")) for n in all_nodes if n.get("id")}
    label_index: dict = {}
    for n in all_nodes:
        label = str(n.get("label") or "")
        if not label or label.startswith(".") or not n.get("id"):
            continue
        sp = source_path(str(n.get("source_file") or ""), root)
        if sp is not None:
            label_index.setdefault((Path(sp), label), str(n["id"]))
    res = _Resolver(node_ids, label_index, parse, resolve_module)
    _mark_escapes(res, paths, nodes, root, source_path)
    if not candidates:
        return
    existing = {(str(e.get("source")), str(e.get("target"))) for e in edges if e.get("relation") == "calls"}
    owned = {str(n.get("id")) for n in nodes}

    plans = []
    def _order(c):
        loc = str(c.get("source_location") or "")
        line = int(loc[1:]) if loc[1:].isdigit() else 0
        return (str(c.get("source_file")), line, str(c.get("receiver")), str(c.get("callee")))

    for call in sorted(candidates, key=_order):
        try:
            if str(call.get("caller_nid")) not in owned:
                continue
            sp = source_path(str(call["source_file"]), root)
            mod = res.module(sp) if sp is not None else None
            if mod is None:
                continue
            line = int(str(call.get("source_location", "L0")).removeprefix("L"))
            sites = []
            for n in mod.calls_at(line):
                fn = n.child_by_field_name("function")
                if fn is None or fn.type != "member_expression":
                    continue
                obj = fn.child_by_field_name("object")
                prop = fn.child_by_field_name("property")
                if (obj is not None and obj.type == "identifier" and prop is not None
                        and _read_text(obj, mod.source) == call["receiver"]
                        and _read_text(prop, mod.source) == call["callee"]
                        and _inside_function(n)
                        and not _is_import_receiver(res, obj, mod)):
                    sites.append(n)
            if sites:
                plans.append((call, mod, sites))
        except Exception:
            continue
    if not plans:
        return

    # Scan the module of every owner a receiver can reach, and every module that
    # imports it, before reading any target: scans only ever close objects off.
    # Closing an object off only ever removes owners, so one discovery pass
    # finds every module a scan may have to cover.
    pending: set = set()
    for _, mod, sites in plans:
        for site in sites:
            for path in _owner_modules(res, site, mod):
                pending.add(path)
                pending.update(Path(p) for p in importers_of(path))
    for path in sorted(pending, key=str):
        _scan_file(res, path)

    emitted: dict = {}
    for call, mod, sites in plans:
        for site in sites:
            targets = _site_targets(res, site, mod)
            if not targets:
                continue
            score = 0.85 if len(targets) == 1 else 0.75
            caller = str(call["caller_nid"])
            for nid in sorted(targets):
                pair = (caller, nid)
                if nid == caller or pair in existing or pair in emitted:
                    continue
                emitted[pair] = {
                    "source": caller,
                    "target": nid,
                    "relation": "calls",
                    "context": CONTEXT,
                    "confidence": "INFERRED",
                    "confidence_score": score,
                    "source_file": call["source_file"],
                    "source_location": call["source_location"],
                    "weight": 1.0,
                }
    edges.extend(emitted[k] for k in sorted(emitted))


ESCAPE_MARKER = "_js_dispatch_escapes"


def _mark_escapes(res, paths, nodes, root, source_path) -> None:
    """Record on each fresh JS file node the other modules whose objects the file
    writes or hands to unknown code. An incremental update reads the persisted
    marker to know which importers a change can affect (the old content is gone)."""
    file_nodes = {}
    for n in nodes:
        sp = source_path(str(n.get("source_file") or ""), root)
        if sp is not None and str(n.get("label") or "") == Path(sp).name:
            file_nodes.setdefault(Path(sp), n)
    for path in sorted({Path(p).resolve() for p in paths}, key=str):
        if effective_suffix(path) not in _JS_SUFFIXES:
            continue
        mod = res.module(path)
        node = file_nodes.get(path)
        if mod is None or node is None or not _imports_function_holders(res, mod):
            continue
        hits = _scan_file(res, path)
        if hits:
            node[ESCAPE_MARKER] = sorted(_portable(h, root) for h in hits)


def _portable(path, root) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _is_import_receiver(res, ident, mod) -> bool:
    """`snack.error()` on an imported object binding is a member call on an
    imported object, not dispatch through a selected config: out of this pass's
    scope (a receiver reached through a local alias or a lookup still resolves)."""
    binding = res.lookup(ident, mod)
    return binding is not None and binding[0] == "import"


def _inside_function(node) -> bool:
    """Module top-level code runs during initialisation, before later top-level
    writes: the final-heap model does not describe it."""
    p = node.parent
    while p is not None:
        if p.type in _FUNC_SCOPES:
            return True
        p = p.parent
    return False


def _receiver_value(res, site, mod):
    fn = site.child_by_field_name("function")
    return res.eval(fn.child_by_field_name("object"), mod)


def _owner_modules(res, site, mod):
    try:
        value = _receiver_value(res, site, mod)
    except (_Top, RecursionError):
        return set()
    out, seen = set(), set()
    stack = [a[1] for a in value if a[0] == "obj"]
    while stack:
        obj = stack.pop()
        if obj.key in seen:
            continue
        seen.add(obj.key)
        out.add(obj.module.path)
        for v in obj.props.values():
            stack.extend(a[1] for a in v or () if a[0] == "obj")
    return out


def _site_targets(res, site, mod):
    """Target node ids for one call site; empty when it stays unresolved."""
    try:
        fn = site.child_by_field_name("function")
        key = _read_text(fn.child_by_field_name("property"), mod.source)
        if key in _OBJECT_PROTOTYPE_KEYS:
            return set()
        receiver = _receiver_value(res, site, mod)
        facts = _site_facts(res, site, mod)
        owners = [a for a in receiver if not _refuted(a[2], facts)]
        if not owners or any(a[0] != "obj" for a in owners):
            return set()
        if len({a[1].key for a in owners}) > MAX_OWNERS:
            return set()
        targets = set()
        for _, obj, conds in owners:
            if not obj.closed:
                return set()
            if key not in obj.props:
                continue
            value = obj.props[key]
            if value is None:
                return set()
            for kind, payload, tconds in _with_conds(value, conds):
                if _refuted(tconds, facts):
                    continue
                if kind == "nonfn":
                    continue
                if kind != "fn" or payload[1]:
                    return set()  # no graph node, a nested object, or a `this` writer
                targets.add(payload[0])
        if len(targets) > MAX_TARGETS:
            return set()
        return targets
    except (_Top, RecursionError):
        return set()
    except Exception:
        return set()


# ── incremental: dependent-file refresh ─────────────────────────────────

def _holds_functions(module_path: Path, parse) -> bool:
    """The module declares a module-level object literal that can hold a
    function (a dispatch owner candidate)."""
    parsed = parse(module_path)
    if parsed is None:
        return False
    return _module_holds_functions(_Module(Path(module_path), parsed[0], parsed[1]))


def dispatch_refresh_paths(changed, deleted, code_files, graph_path: Path, roots, parse, resolve_module):
    """Unchanged JS/TS files an incremental update must re-extract because their
    config-dispatch edges depend on a changed or deleted module.

    A dispatch edge belongs to the calling file but its targets come from the
    owner module, so re-extracting only the owner would leave the caller's edges
    stale (or miss new ones). Re-extract every unchanged file that
      (a) has a persisted ``config_dispatch`` edge into a changed/deleted module;
      (b) imports a changed module that can own dispatch targets; or
      (c) imports a module that a changed file imports (before or after the
          change) and that can own dispatch targets — the changed file may now
          write to, or hand off, those owners.
    """
    import json
    import os

    def is_js(p) -> bool:
        return effective_suffix(str(p)) in _JS_SUFFIXES

    js_changed = [Path(os.path.abspath(p)) for p in changed if is_js(p)]
    js_deleted = [Path(os.path.abspath(p)) for p in deleted if is_js(p)]
    if not (js_changed or js_deleted) or not Path(graph_path).is_file():
        return []
    try:
        data = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    except Exception:
        return []

    def absolute(source_file):
        if not source_file:
            return None
        p = Path(source_file)
        if p.is_absolute():
            return Path(os.path.abspath(p))
        for root in roots:
            candidate = Path(os.path.abspath(Path(root) / p))
            if candidate.exists():
                return candidate
        return Path(os.path.abspath(Path(roots[0]) / p))

    node_file = {str(n.get("id")): absolute(n.get("source_file")) for n in data.get("nodes", [])}
    links = data.get("links", data.get("edges", []))
    changed_set = set(js_changed) | set(js_deleted)
    relevance: dict[Path, bool] = {}

    def relevant(path) -> bool:
        if path is None or not is_js(path):
            return False
        if path not in relevance:
            relevance[path] = path.is_file() and _holds_functions(path, parse)
        return relevance[path]

    add: set[Path] = set()
    owners: set[Path] = set()
    for link in links:
        if link.get("context") == CONTEXT and node_file.get(str(link.get("target"))) in changed_set:
            add.add(absolute(link.get("source_file")))
    for path in changed_set:
        if relevant(path):
            owners.add(path)
    # (c) a change that starts or stops writing to / handing off another
    # module's objects: before, from the persisted marker; after, by scanning.
    for n in data.get("nodes", []):
        if absolute(n.get("source_file")) in changed_set:
            for rel in n.get(ESCAPE_MARKER) or ():
                target = absolute(rel)
                if target is not None:
                    owners.add(target)
    scanner = _Resolver(set(), {}, parse, resolve_module)
    for path in js_changed:
        if path.is_file():
            try:
                owners.update(Path(os.path.abspath(h)) for h in _scan_file(scanner, path.resolve()))
            except Exception:
                return [p for p in code_files if is_js(p) and Path(os.path.abspath(p)) not in changed_set]
    if owners:
        for link in links:
            if link.get("relation") in ("imports", "imports_from") \
                    and node_file.get(str(link.get("target"))) in owners:
                add.add(absolute(link.get("source_file")))
    code = {Path(os.path.abspath(p)): p for p in code_files if is_js(p)}
    return [code[p] for p in sorted(add - changed_set, key=str) if p in code]
