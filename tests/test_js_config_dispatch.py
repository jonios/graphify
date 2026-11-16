"""Owner-traced config dispatch (`config.submit(p)`, `cfg.modes[mode].run()`).

A member call through a configuration object calls whatever function the object
holds. The receiver is traced, through a small set of statically understood
constructs, to the objects it can be (identified by where they are allocated);
the call then gets an INFERRED `calls` edge (`context: "config_dispatch"`) to
every function those owners hold under the called key — at most four — or none
at all when any alternative is unknown. Property names alone never resolve.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from graphify.extract import extract
from graphify.watch import _rebuild_code

CTX = "config_dispatch"


def _write(root: Path, files: dict[str, str]) -> list[Path]:
    out = []
    for name, body in files.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        out.append(p)
    return out


def _extract(tmp_path: Path, files: dict[str, str]):
    return extract(_write(tmp_path, files), cache_root=tmp_path / "graphify-out",
                   root=tmp_path, parallel=False)


def _names(r) -> dict[str, str]:
    """Node id -> `Owner.key` for object members, else the bare label."""
    label = {n["id"]: str(n["label"]) for n in r["nodes"]}
    owner = {e["target"]: label.get(e["source"]) for e in r["edges"] if e["relation"] == "method"}
    out = {}
    for nid, lab in label.items():
        bare = lab.strip("()").lstrip(".")
        out[nid] = f"{owner[nid]}.{bare}" if lab.startswith(".") and nid in owner else bare
    return out


def _dispatch(r) -> list[tuple[str, str, float]]:
    names = _names(r)
    return sorted((names.get(e["source"], "?"), names.get(e["target"], "?"), e["confidence_score"])
                  for e in r["edges"] if e.get("context") == CTX)


def _edges(r):
    return [e for e in r["edges"] if e.get("context") == CTX]


# ── Supported owner traces ──────────────────────────────────────────────────

def test_direct_object_arrow_function_and_shorthand_members(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "function save() { return 1; }\n"
        "export const CONFIG = {\n"
        "  arrow: (p) => p,\n"
        "  fnExpr: function (p) { return p; },\n"
        "  short(p) { return p; },\n"
        "  ref: save,\n"
        "};\n"
        "export function run(p) {\n"
        "  CONFIG.arrow(p); CONFIG.fnExpr(p); CONFIG.short(p); CONFIG.ref(p);\n"
        "}\n")})
    assert _dispatch(r) == [
        ("run", "CONFIG.arrow", 0.85), ("run", "CONFIG.fnExpr", 0.85),
        ("run", "CONFIG.short", 0.85), ("run", "save", 0.85),
    ]
    for e in _edges(r):
        assert e["relation"] == "calls" and e["confidence"] == "INFERRED"


def test_local_alias_and_alias_chain(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "export const CONFIG = { submit: (p) => p };\n"
        "export function one(p) { const c = CONFIG; return c.submit(p); }\n"
        "export function two(p) { const a = CONFIG; const b = a; return b.submit(p); }\n")})
    assert _dispatch(r) == [("one", "CONFIG.submit", 0.85), ("two", "CONFIG.submit", 0.85)]


def test_imported_config_through_a_barrel(tmp_path):
    r = _extract(tmp_path, {
        "cfg/config.js": "export const CONFIG = { submit: (p) => p };\n",
        "cfg/index.js": "export { CONFIG } from './config';\n",
        "app.js": ("import { CONFIG } from './cfg';\n"
                   "export function save(p) { return CONFIG.submit(p); }\n"),
    })
    assert _dispatch(r) == [("save", "CONFIG.submit", 0.85)]


def test_lookup_with_one_return(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "const A = { submit: (p) => p };\n"
        "export function getConfig() { return A; }\n"
        "export function save(p) { const c = getConfig(); return c.submit(p); }\n")})
    assert _dispatch(r) == [("save", "A.submit", 0.85)]


def test_branches_reaching_the_same_function_give_one_edge(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "function shared(p) { return p; }\n"
        "const A = { submit: shared };\n"
        "const B = { submit: shared };\n"
        "export function getConfig(t) { if (t) { return A; } return B; }\n"
        "export function save(t, p) { const c = getConfig(t); return c.submit(p); }\n")})
    assert _dispatch(r) == [("save", "shared", 0.85)]


def test_branches_reaching_different_functions_give_one_edge_each(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "const A = { submit: (p) => p };\n"
        "const B = { submit: (p) => p };\n"
        "export function getConfig(t) { if (t === 'a') { return A; } return B; }\n"
        "export function save(t, p) { const c = getConfig(t); return c.submit(p); }\n")})
    assert _dispatch(r) == [("save", "A.submit", 0.75), ("save", "B.submit", 0.75)]


def test_table_lookup_with_default_and_nested_mode_path(tmp_path):
    r = _extract(tmp_path, {
        "cfg.js": (
            "export const TABLE = {\n"
            "  Pickup: { modes: { bulk: { submit: (p) => p }, single: { submit: (p) => p } } },\n"
            "  Ondemand: { modes: { single: { submit: (p) => p } } },\n"
            "};\n"
            "export const getConfig = (type) => TABLE[type] || TABLE.Pickup;\n"),
        "app.js": (
            "import { getConfig } from './cfg';\n"
            "export function save(type, mode, p) {\n"
            "  const config = getConfig(type);\n"
            "  const modeConfig = config.modes[mode];\n"
            "  return modeConfig.submit(p);\n"
            "}\n"),
    })
    # bulk/single members of one owner share one node (Stage 2's collapse).
    assert _dispatch(r) == [("save", "TABLE.submit", 0.85)]


def test_static_computed_key_selects_one_owner(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "const A = { run: (x) => x };\n"
        "const B = { run: (x) => x };\n"
        "export const T = { a: A, b: B };\n"
        "export function go(x) { const c = T['a']; return c.run(x); }\n")})
    assert _dispatch(r) == [("go", "A.run", 0.85)]


def test_spread_object_assign_and_top_level_factory_assignment(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "function base(p) { return p; }\n"
        "function other(p) { return p; }\n"
        "const PARTS = { Ondemand: { submit: base, close: base } };\n"
        "const make = () => ({ ...PARTS.Ondemand, close: other });\n"
        "PARTS.Pickup = make();\n"
        "export const TABLE = { Delivery: { submit: other } };\n"
        "Object.assign(TABLE, PARTS);\n"
        "export const lookup = (t) => TABLE[t] || TABLE.Delivery;\n"
        "export function save(t, p) { const c = lookup(t); c.submit(p); return c.close(p); }\n")})
    assert _dispatch(r) == [("save", "base", 0.75), ("save", "other", 0.75)]


def test_optional_calls_and_absent_key_on_a_closed_owner(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "const A = { onClose: (d) => d };\n"
        "const B = {};\n"
        "export const T = { a: A, b: B };\n"
        "export function close(t, d) { const c = T[t]; c.onClose?.(d); c?.onClose(d); }\n")})
    assert _dispatch(r) == [("close", "A.onClose", 0.85)]


def test_top_level_mutation_before_dispatch_is_followed(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "function first(p) { return p; }\n"
        "function second(p) { return p; }\n"
        "export const CONFIG = { submit: first };\n"
        "CONFIG.submit = second;\n"
        "export function save(p) { return CONFIG.submit(p); }\n")})
    assert _dispatch(r) == [("save", "second", 0.85)]


def test_cyclic_object_reference_terminates(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "export const CONFIG = { run: (x) => x };\n"
        "CONFIG.self = CONFIG;\n"
        "export function go(x) { const c = CONFIG.self.self; return c.run(x); }\n")})
    assert _dispatch(r) == [("go", "CONFIG.run", 0.85)]


# ── Branch conditions ──────────────────────────────────────────────────────

def test_guard_excludes_an_owner_selected_by_key(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "const A = { run: (x) => x };\n"
        "const B = { run: (x) => x };\n"
        "export const T = { a: A, b: B };\n"
        "export function go(kind, x) {\n"
        "  const isB = kind === 'b';\n"
        "  if (!isB) return null;\n"
        "  const c = T[kind];\n"
        "  return c.run(x);\n"
        "}\n")})
    assert _dispatch(r) == [("go", "B.run", 0.85)]


def test_guard_keeps_the_fallback_reachable_for_unknown_keys(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "const A = { run: (x) => x };\n"
        "const B = { run: (x) => x };\n"
        "export const T = { a: A, b: B };\n"
        "export function go(kind, x) {\n"
        "  if (kind !== 'a') {\n"
        "    const c = T[kind] || T.a;\n"
        "    return c.run(x);\n"
        "  }\n"
        "  return null;\n"
        "}\n"
        "export function only(kind, x) {\n"
        "  if (kind === 'b') { const c = T[kind] || T.a; return c.run(x); }\n"
        "  return null;\n"
        "}\n")})
    # `kind !== 'a'` removes A selected by key, but an unknown kind still falls
    # back to T.a; `kind === 'b'` proves the fallback unreachable.
    assert _dispatch(r) == [("go", "A.run", 0.75), ("go", "B.run", 0.75), ("only", "B.run", 0.85)]


# ── Unresolved by design ───────────────────────────────────────────────────

def _assert_none(tmp_path, files):
    r = _extract(tmp_path, files)
    assert _dispatch(r) == []
    return r


def test_unknown_return_branch(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "const A = { submit: (p) => p };\n"
        "export function getConfig(t) { if (t) { return A; } return fetchConfig(t); }\n"
        "export function save(t, p) { const c = getConfig(t); return c.submit(p); }\n")})


def test_factory_local_functions_have_no_node(tmp_path):
    # The logger shape: the trace exists, but the target has no graph node.
    _assert_none(tmp_path, {
        "logger.js": ("export function createLogger(opts) {\n"
                      "  return { error: (m) => m, warn: (m) => m };\n"
                      "}\n"),
        "app.js": ("import { createLogger } from './logger';\n"
                   "export function run() { const logger = createLogger({}); logger.error('x'); }\n"),
    })


def test_dynamic_key_over_an_open_object(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "const A = { run: (x) => x };\n"
        "export const T = { a: A };\n"
        "export function register(k, v) { T[k] = v; }\n"
        "export function go(k, x) { const c = T[k]; return c.run(x); }\n")})


def test_computed_literal_key_and_getter(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "const name = 'run';\n"
        "export const A = { [name]: (x) => x, other: (x) => x };\n"
        "export const B = { get run() { return (x) => x; } };\n"
        "export function go(x) { A.other(x); return B.run(x); }\n")})


def test_parameter_receivers_stay_unresolved(tmp_path):
    # Pattern E is deferred: even a parameter whose only caller passes CONFIG.
    _assert_none(tmp_path, {"cfg.js": (
        "export const CONFIG = { submit: (p) => p };\n"
        "export const OTHER = { submit: (p) => p };\n"
        "function useIt(cfg, p) { return cfg.submit(p); }\n"
        "function conflicting(cfg, p) { return cfg.submit(p); }\n"
        "export function a(p) { useIt(CONFIG, p); conflicting(CONFIG, p); }\n"
        "export function b(p) { conflicting(OTHER, p); }\n")})


def test_local_variable_shadowing_an_imported_config(tmp_path):
    _assert_none(tmp_path, {
        "cfg.js": "export const CONFIG = { submit: (p) => p };\n",
        "app.js": ("import { CONFIG } from './cfg';\n"
                   "export function save(p, other) { const CONFIG = other; return CONFIG.submit(p); }\n"
                   "export function save2(CONFIG, p) { return CONFIG.submit(p); }\n"),
    })


def test_mutation_inside_a_function_and_unknown_assign_source(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export const A = { submit: (p) => p };\n"
        "export const B = { submit: (p) => p };\n"
        "Object.assign(B, loadOverrides());\n"
        "export function patch(fn) { A.submit = fn; }\n"
        "export function save(p) { A.submit(p); return B.submit(p); }\n")})


def test_top_level_mutation_with_an_unknown_value(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export const A = { submit: (p) => p };\n"
        "A.submit = makeHandler();\n"
        "export function save(p) { return A.submit(p); }\n")})


def test_delete_and_owner_passed_to_unknown_code(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export const A = { submit: (p) => p };\n"
        "export const B = { submit: (p) => p };\n"
        "delete A.extra;\n"
        "register(B);\n"
        "export function save(p) { A.submit(p); return B.submit(p); }\n")})


def test_write_through_the_lookup_result_in_another_importer(tmp_path):
    _assert_none(tmp_path, {
        "cfg.js": ("const A = { submit: (p) => p };\n"
                   "export const getConfig = () => A;\n"),
        "app.js": ("import { getConfig } from './cfg';\n"
                   "export function save(p) { const c = getConfig(); return c.submit(p); }\n"),
        "evil.js": ("import { getConfig } from './cfg';\n"
                    "export function patch(fn) { const c = getConfig(); c.submit = fn; }\n"),
    })


def test_write_through_an_alias(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export const A = { submit: (p) => p };\n"
        "export function patch(fn) { const alias = A; alias.submit = fn; }\n"
        "export function save(p) { return A.submit(p); }\n")})


def test_recursive_alias_terminates_unresolved(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export function go(x) { const a = b; const b = a; return a.run(x); }\n")})


def test_more_than_four_targets(tmp_path):
    entries = "\n".join(f"  k{i}: {{ run: function f{i}(x) {{ return x; }} }}," for i in range(5))
    r = _extract(tmp_path, {"cfg.js": (
        "function f0(x) { return x; }\nfunction f1(x) { return x; }\nfunction f2(x) { return x; }\n"
        "function f3(x) { return x; }\nfunction f4(x) { return x; }\n"
        "export const T = { k0: { run: f0 }, k1: { run: f1 }, k2: { run: f2 }, k3: { run: f3 }, k4: { run: f4 } };\n"
        "export function go(k, x) { const c = T[k]; return c.run(x); }\n")})
    assert entries  # five distinct targets: over the bound, so none
    assert _dispatch(r) == []


def test_prototype_key_and_top_level_dispatch(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export const T = { a: { run: (x) => x } };\n"
        "export function go(k) { const c = T[k]; return c.toString(); }\n"
        "T.a.run(1);\n")})


def test_unrelated_same_name_receivers(tmp_path):
    # breaker / logger / params / mw shapes next to a config that has the keys.
    _assert_none(tmp_path, {
        "cfg.js": ("export const CONFIG = { onSuccess: (d) => d, error: (e) => e,\n"
                   "  success: (d) => d, onError: (e) => e };\n"),
        "app.js": (
            "import './cfg';\n"
            "export function a(rt, ctx) { const breaker = rt.circuits.get(ctx); breaker.onSuccess(1); }\n"
            "export function b(params) { params.success({}); }\n"
            "export function c(chain, i) { const mw = chain[i]; return mw.onError(1); }\n"
            "export function d(make) { const logger = make(); logger.error('x'); }\n"
            "export function e(x) { const inst = new Thing(); inst.onSuccess(x); }\n"
            "export function f(list) { const first = list[0]; first.onError(1); }\n"
            "export function g(x) { unknownGlobal.onSuccess(x); }\n"),
    })


def test_same_named_methods_on_unrelated_objects(tmp_path):
    r = _extract(tmp_path, {
        "a.js": "export const A = { submit: (p) => p };\n",
        "b.js": ("export const B = { submit: (p) => p };\n"
                 "export function save(p) { return B.submit(p); }\n"),
    })
    assert _dispatch(r) == [("save", "B.submit", 0.85)]


def test_this_writing_method_is_not_resolved(tmp_path):
    _assert_none(tmp_path, {"cfg.js": (
        "export const A = { submit(p) { this.last = p; return p; } };\n"
        "export function save(p) { return A.submit(p); }\n")})


# ── Invariants ─────────────────────────────────────────────────────────────

def test_no_self_loop_and_no_duplicate_edges(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "export const CONFIG = { run(x) { return x; }, again(x) { return CONFIG.again(x); } };\n"
        "export function go(x) { CONFIG.run(x); CONFIG.run(x); return CONFIG.run(x); }\n")})
    edges = _edges(r)
    assert all(e["source"] != e["target"] for e in edges)
    assert _dispatch(r) == [("go", "CONFIG.run", 0.85)]


def test_existing_extracted_edge_is_not_duplicated(tmp_path):
    r = _extract(tmp_path, {"cfg.js": (
        "export function save(p) { return p; }\n"
        "export const CONFIG = { save };\n"
        "export function go(p) { save(p); return CONFIG.save(p); }\n")})
    pairs = [(e["source"], e["target"]) for e in r["edges"] if e["relation"] == "calls"]
    assert len(pairs) == len(set(pairs))
    assert _dispatch(r) == []  # the direct EXTRACTED call already links go -> save


def test_deterministic_output(tmp_path):
    files = {"cfg.js": (
        "const A = { submit: (p) => p };\nconst B = { submit: (p) => p };\n"
        "export const T = { a: A, b: B };\n"
        "export function save(t, p) { const c = T[t]; return c.submit(p); }\n")}
    first = _edges(_extract(tmp_path / "one", files))
    second = _edges(_extract(tmp_path / "two", files))
    assert [(e["source"], e["target"], e["confidence_score"]) for e in first] == \
        [(e["source"], e["target"], e["confidence_score"]) for e in second]
    assert len(first) == 2


# ── Incremental updates equal a fresh full build ───────────────────────────

_PROJECT = {
    "cfg.js": (
        "function sharedPrepare(d) { return d; }\n"
        "const PARTS = { Ondemand: { prepare: sharedPrepare, modes: { single: { submit: (p) => p } } } };\n"
        "const make = () => ({ ...PARTS.Ondemand });\n"
        "PARTS.Pickup = make();\n"
        "export const TABLE = { Delivery: { prepare: sharedPrepare, modes: { bulk: { submit: (p) => p } } } };\n"
        "Object.assign(TABLE, PARTS);\n"
        "export const getConfig = (type) => TABLE[type] || TABLE.Delivery;\n"),
    "app.js": (
        "import { getConfig } from './cfg';\n"
        "export function Modal({ type, mode }) {\n"
        "  const config = getConfig(type);\n"
        "  const modeConfig = config.modes[mode];\n"
        "  config.prepare(1);\n"
        "  return modeConfig.submit(2);\n"
        "}\n"),
    "other.js": "export function unrelated() { return 1; }\n",
}


def _graph_dispatch(root: Path):
    data = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    return sorted((l["source"], l["target"], l["relation"], l.get("confidence"),
                   l.get("confidence_score"), l.get("context"))
                  for l in data["links"] if l.get("context") == CTX)


def _graph_calls(root: Path):
    data = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    return sorted((l["source"], l["target"], l["relation"], l.get("confidence"), l.get("context"))
                  for l in data["links"] if l["relation"] == "calls")


def _check_incremental(tmp_path: Path, edits: dict[str, str]):
    inc = tmp_path / "inc"
    _write(inc, _PROJECT)
    assert _rebuild_code(inc, no_cluster=True)
    before = _graph_dispatch(inc)
    for name, body in edits.items():
        (inc / name).write_text(body, encoding="utf-8")
    assert _rebuild_code(inc, incremental=True, no_cluster=True)
    full = tmp_path / "full"
    shutil.copytree(inc, full, ignore=shutil.ignore_patterns("graphify-out"))
    assert _rebuild_code(full, no_cluster=True)
    assert _graph_dispatch(inc) == _graph_dispatch(full)
    assert _graph_calls(inc) == _graph_calls(full)
    return before, _graph_dispatch(inc)


def test_initial_build_resolves_the_project(tmp_path):
    root = tmp_path / "p"
    _write(root, _PROJECT)
    assert _rebuild_code(root, no_cluster=True)
    targets = {t for _, t, *_ in _graph_dispatch(root)}
    assert len(targets) == 3  # sharedPrepare + two collapsed `submit` members


def test_incremental_after_changing_a_config_value(tmp_path):
    cfg = _PROJECT["cfg.js"].replace(
        "export const TABLE = { Delivery: { prepare: sharedPrepare,",
        "function deliveryPrepare(d) { return d; }\n"
        "export const TABLE = { Delivery: { prepare: deliveryPrepare,")
    before, after = _check_incremental(tmp_path, {"cfg.js": cfg})
    assert before != after


def test_incremental_after_changing_the_factory_return(tmp_path):
    cfg = _PROJECT["cfg.js"].replace(
        "const make = () => ({ ...PARTS.Ondemand });",
        "function pickupPrepare(d) { return d; }\n"
        "const make = () => ({ ...PARTS.Ondemand, prepare: pickupPrepare });")
    before, after = _check_incremental(tmp_path, {"cfg.js": cfg})
    assert before != after


def test_incremental_after_adding_an_alias_in_the_caller(tmp_path):
    app = _PROJECT["app.js"].replace("  return modeConfig.submit(2);",
                                     "  const again = modeConfig;\n  return again.submit(2);")
    _check_incremental(tmp_path, {"app.js": app})


def test_incremental_after_mutating_a_config_property(tmp_path):
    cfg = _PROJECT["cfg.js"] + ("function override(d) { return d; }\n"
                                "TABLE.Delivery.prepare = override;\n")
    before, after = _check_incremental(tmp_path, {"cfg.js": cfg})
    assert before != after


def test_incremental_after_removing_a_relationship(tmp_path):
    cfg = _PROJECT["cfg.js"].replace("prepare: sharedPrepare, modes: { bulk:", "modes: { bulk:")
    cfg = cfg.replace("{ Ondemand: { prepare: sharedPrepare, modes", "{ Ondemand: { modes")
    before, after = _check_incremental(tmp_path, {"cfg.js": cfg})
    assert len(after) < len(before)


def test_incremental_after_another_importer_starts_mutating(tmp_path):
    evil = ("import { getConfig } from './cfg';\n"
            "export function patch(fn) { const c = getConfig('x'); c.prepare = fn; }\n")
    before, after = _check_incremental(tmp_path, {"other.js": evil})
    assert before and not any(t.endswith("sharedprepare") for _, t, *_ in after)


def test_incremental_unchanged_caller_becomes_resolvable(tmp_path):
    inc = tmp_path / "inc"
    project = dict(_PROJECT)
    project["cfg.js"] = _PROJECT["cfg.js"].replace(
        "export const getConfig = (type) => TABLE[type] || TABLE.Delivery;",
        "export const getConfig = (type) => TABLE[type] || loadDefault();")
    _write(inc, project)
    assert _rebuild_code(inc, no_cluster=True)
    assert _graph_dispatch(inc) == []
    (inc / "cfg.js").write_text(_PROJECT["cfg.js"], encoding="utf-8")
    assert _rebuild_code(inc, incremental=True, no_cluster=True)
    full = tmp_path / "full"
    shutil.copytree(inc, full, ignore=shutil.ignore_patterns("graphify-out"))
    assert _rebuild_code(full, no_cluster=True)
    assert _graph_dispatch(inc) == _graph_dispatch(full) != []


def test_partly_unlinkable_alternatives_emit_nothing(tmp_path):
    # One owner holds a linkable member, the other a factory-local function: the
    # set is incomplete, so not even the linkable half is emitted.
    _assert_none(tmp_path, {"cfg.js": (
        "const make = () => ({ run: (x) => x });\n"
        "export const T = { a: { run: (x) => x } };\n"
        "T.b = make();\n"
        "export function go(k, x) { const c = T[k]; return c.run(x); }\n")})


def test_top_level_dispatch_runs_before_later_writes(tmp_path):
    # At module top level `T.run(1)` executes during initialisation and calls
    # `first`; the final heap (after `T.run = second`) does not describe it.
    # The engine records no raw calls for top-level code today, so the resolver
    # is driven directly with one.
    from graphify.extractors.js_config_dispatch import resolve_config_dispatch
    from graphify.extractors.resolution import (
        _js_source_path, _parse_js_tree, _resolve_js_module_path,
    )
    r = _extract(tmp_path, {"cfg.js": (
        "function first(x) { return x; }\n"
        "function second(x) { return x; }\n"
        "export const T = { run: first };\n"
        "T.run(1);\n"
        "T.run = second;\n"
        "export function later(x) { return T.run(x); }\n")})
    file_node = next(n["id"] for n in r["nodes"] if n["label"] == "cfg.js")
    later = next(n["id"] for n in r["nodes"] if n["label"] == "later()")
    source = str(tmp_path / "cfg.js")
    raw = [
        {"caller_nid": file_node, "callee": "run", "is_member_call": True, "receiver": "T",
         "source_file": source, "source_location": "L4"},
        {"caller_nid": later, "callee": "run", "is_member_call": True, "receiver": "T",
         "source_file": source, "source_location": "L6"},
    ]
    edges: list[dict] = []
    resolve_config_dispatch([tmp_path / "cfg.js"], r["nodes"], edges, tmp_path, raw, None,
                            lambda p: set(), _parse_js_tree, _resolve_js_module_path, _js_source_path)
    names = _names(r)
    assert [(names[e["source"]], names[e["target"]]) for e in edges] == [("later", "second")]


def test_incremental_after_an_importer_stops_mutating(tmp_path):
    # The old content is gone by the time the update runs: the persisted escape
    # marker on the changed file says which importers to refresh.
    inc = tmp_path / "inc"
    project = dict(_PROJECT)
    project["other.js"] = ("import { getConfig } from './cfg';\n"
                           "export function patch(fn) { const c = getConfig('x'); c.prepare = fn; }\n")
    _write(inc, project)
    assert _rebuild_code(inc, no_cluster=True)
    assert not any(t.endswith("sharedprepare") for _, t, *_ in _graph_dispatch(inc))
    nodes = json.loads((inc / "graphify-out" / "graph.json").read_text(encoding="utf-8"))["nodes"]
    marker = [n.get("_js_dispatch_escapes") for n in nodes if n.get("source_file") == "other.js"
              and n.get("label") == "other.js"]
    assert marker == [["cfg.js"]]
    (inc / "other.js").write_text(_PROJECT["other.js"], encoding="utf-8")
    assert _rebuild_code(inc, incremental=True, no_cluster=True)
    full = tmp_path / "full"
    shutil.copytree(inc, full, ignore=shutil.ignore_patterns("graphify-out"))
    assert _rebuild_code(full, no_cluster=True)
    assert _graph_dispatch(inc) == _graph_dispatch(full)
    assert any(t.endswith("sharedprepare") for _, t, *_ in _graph_dispatch(inc))


def test_refresh_is_limited_to_real_dependents(tmp_path):
    # A file that only reads another module's config adds no dependents; one
    # that writes to it pulls in that module's importers.
    from graphify.extractors.js_config_dispatch import dispatch_refresh_paths
    from graphify.extractors.resolution import _parse_js_tree, _resolve_js_module_path
    root = tmp_path / "p"
    project = dict(_PROJECT)
    project["reader.js"] = ("import { getConfig } from './cfg';\n"
                            "export function read(t) { return getConfig(t).prepare(1); }\n")
    files = _write(root, project)
    assert _rebuild_code(root, no_cluster=True)
    graph = root / "graphify-out" / "graph.json"

    def refresh(changed):
        return sorted(p.name for p in dispatch_refresh_paths(
            [root / changed], [], files, graph, (root,), _parse_js_tree, _resolve_js_module_path))

    assert refresh("reader.js") == []
    assert refresh("other.js") == []
    assert refresh("cfg.js") == ["app.js", "reader.js"]
    (root / "reader.js").write_text(
        "import { getConfig } from './cfg';\n"
        "export function read(t, fn) { getConfig(t).prepare = fn; }\n", encoding="utf-8")
    assert refresh("reader.js") == ["app.js"]
