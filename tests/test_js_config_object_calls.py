"""Calls inside module-level JS/TS configuration objects.

`const CONFIG = { bulk: { submit: (p) => api(p) } }` produced only the `CONFIG`
node: the module-level-const branch of `_js_extra_walk` returns before the main
walk can descend, and it tracked closures only for call/new initializers
(#2552), so every function in an object or array literal was dropped with its
calls (#2419).

Function-valued properties (arrow, function expression, shorthand method) of a
module-level object/array literal now become `.key()` member nodes owned by the
binding through a `method` edge (the #2433 / exported-object convention), nested
objects and arrays flattened onto the binding. Same-named members collapse into
one node, but every body is still walked. Closures with no property key (array
elements, call arguments) are walked under the binding itself, as #2552 does for
call initializers. Dynamic dispatch through the object stays unresolved.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_PAYLOAD = "export function buildPayload(ctx) { return ctx; }\nexport function helper() { return 1; }\n"
_ORDER_API = "export class OrderApi {\n  BatchUpdateStatusV2(p) { return p; }\n}\n"


def _extract(tmp_path: Path, files: dict[str, str]):
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return extract([tmp_path / n for n in files],
                   cache_root=tmp_path / "graphify-out", parallel=False)


def _labels(r) -> list[str]:
    return [n["label"] for n in r["nodes"]]


def _calls(r) -> list[tuple[str, str]]:
    lbl = {n["id"]: n["label"] for n in r["nodes"]}
    return [(lbl.get(e["source"]), lbl.get(e["target"])) for e in r["edges"]
            if e["relation"] == "calls"]


def _owners(r, member_label: str) -> list[tuple[str, str]]:
    """(owner label, relation) of every structural edge into `member_label`."""
    lbl = {n["id"]: n["label"] for n in r["nodes"]}
    return [(lbl.get(e["source"]), e["relation"]) for e in r["edges"]
            if lbl.get(e["target"]) == member_label
            and e["relation"] in ("method", "contains")]


# ── Function-valued properties of module-level objects ───────────────────────

def test_arrow_property_in_exported_config_calls_imported_function(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload } from './payload';\n"
                      "export const CONFIG = {\n"
                      "  build: (ctx) => buildPayload(ctx),\n"
                      "};\n"),
    })
    assert _owners(r, ".build()") == [("CONFIG", "method")]
    assert (".build()", "buildPayload()") in _calls(r)


def test_function_expression_and_shorthand_method_properties(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload, helper } from './payload';\n"
                      "const CONFIG = {\n"
                      "  viaFunction: function (ctx) { return buildPayload(ctx); },\n"
                      "  viaShorthand(ctx) { return helper(ctx); },\n"
                      "  async viaAsync() { return helper(); },\n"
                      "};\n"
                      "export default CONFIG;\n"),
    })
    calls = _calls(r)
    assert (".viaFunction()", "buildPayload()") in calls
    assert (".viaShorthand()", "helper()") in calls
    assert (".viaAsync()", "helper()") in calls
    for m in (".viaFunction()", ".viaShorthand()", ".viaAsync()"):
        assert _owners(r, m) == [("CONFIG", "method")], m


def test_nested_objects_flatten_onto_the_binding(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "api/order.js": _ORDER_API,
        "config.js": ("import { buildPayload } from './payload';\n"
                      "import { OrderApi } from './api/order';\n"
                      "export const EditConfig = {\n"
                      "  PickupDelivery: {\n"
                      "    modes: {\n"
                      "      bulk: {\n"
                      "        buildPayload: (ctx) => buildPayload(ctx),\n"
                      "        submit: (payload) => new OrderApi().BatchUpdateStatusV2(payload),\n"
                      "      },\n"
                      "    },\n"
                      "  },\n"
                      "};\n"),
    })
    calls = _calls(r)
    assert _owners(r, ".submit()") == [("EditConfig", "method")]
    assert (".submit()", "OrderApi") in calls
    assert (".submit()", ".BatchUpdateStatusV2()") in calls  # with the Stage 1 receiver rule
    assert (".buildPayload()", "buildPayload()") in calls


def test_objects_inside_arrays_are_members(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload, helper } from './payload';\n"
                      "export const SECTIONS = [\n"
                      "  { visible: (ctx) => helper(ctx), render: (ctx) => buildPayload(ctx) },\n"
                      "];\n"),
    })
    calls = _calls(r)
    assert (".visible()", "helper()") in calls
    assert (".render()", "buildPayload()") in calls
    assert _owners(r, ".visible()") == [("SECTIONS", "method")]


def test_keyless_closures_are_walked_under_the_binding(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload, helper } from './payload';\n"
                      "export const HANDLERS = [\n"
                      "  (ctx) => buildPayload(ctx),\n"
                      "  { ['computed']: () => helper() },\n"
                      "];\n"),
    })
    calls = _calls(r)
    assert ("HANDLERS", "buildPayload()") in calls
    assert ("HANDLERS", "helper()") in calls
    assert ".computed()" not in _labels(r)


def test_same_named_members_collapse_but_every_body_is_walked(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload, helper } from './payload';\n"
                      "export const CONFIG = {\n"
                      "  bulk: { submit: (p) => buildPayload(p) },\n"
                      "  single: { submit: (p) => helper(p) },\n"
                      "};\n"),
    })
    assert _labels(r).count(".submit()") == 1
    assert _owners(r, ".submit()") == [("CONFIG", "method")]
    calls = _calls(r)
    assert (".submit()", "buildPayload()") in calls
    assert (".submit()", "helper()") in calls


def test_each_closure_is_walked_once(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload } from './payload';\n"
                      "export const CONFIG = {\n"
                      "  outer: (ctx) => [1, 2].map(() => buildPayload(ctx)),\n"
                      "};\n"),
    })
    calls = _calls(r)
    # The nested callback belongs to `.outer()` only: one edge, never also the binding's.
    assert calls.count((".outer()", "buildPayload()")) == 1
    assert ("CONFIG", "buildPayload()") not in calls


def test_typescript_config_object(tmp_path):
    r = _extract(tmp_path, {
        "payload.ts": _PAYLOAD,
        "config.ts": ("import { buildPayload } from './payload';\n"
                      "export const CONFIG = {\n"
                      "  build: (ctx: unknown) => buildPayload(ctx),\n"
                      "} as const;\n"),
    })
    assert (".build()", "buildPayload()") in _calls(r)
    assert _owners(r, ".build()") == [("CONFIG", "method")]


# ── Members are not bare-name targets ────────────────────────────────────────

def test_member_does_not_capture_a_same_named_bare_call(tmp_path):
    # `buildPayload: (ctx) => buildPayload(ctx)` delegates to the import; the
    # call must not bind to the member itself.
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload } from './payload';\n"
                      "export const CONFIG = { buildPayload: (ctx) => buildPayload(ctx) };\n"),
    })
    calls = _calls(r)
    assert (".buildPayload()", "buildPayload()") in calls
    assert (".buildPayload()", ".buildPayload()") not in calls


def test_member_does_not_make_a_cross_file_name_ambiguous(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": "export const CONFIG = { helper: () => 1 };\n",
        "use.js": ("import { helper } from './payload';\n"
                   "export function run() { return helper(); }\n"),
    })
    ids = {n["id"]: n for n in r["nodes"]}
    targets = [ids[e["target"]] for e in r["edges"]
               if e["relation"] == "calls" and ids.get(e["source"], {}).get("label") == "run()"]
    assert [t["label"] for t in targets] == ["helper()"]
    assert targets[0]["source_file"].endswith("payload.js")


def test_member_is_not_an_indirect_call_target_by_bare_name(tmp_path):
    # A bare identifier passed as a value names something in scope; it must
    # not resolve, by label, to an object-literal member here or elsewhere.
    r = _extract(tmp_path, {
        "config.js": "export const CONFIG = { onSuccess: (d) => d, submit: function (p) { return p; } };\n",
        "use.js": ("export function run(xs) { return xs.map(onSuccess); }\n"
                   "export function go(f) { return f(submit); }\n"),
        "same.js": ("export const SAME = { onDone: (d) => d };\n"
                    "export function local(xs) { return xs.forEach(onDone); }\n"),
    })
    lbl = {n["id"]: n["label"] for n in r["nodes"]}
    leaked = [(lbl.get(e["source"]), lbl.get(e["target"])) for e in r["edges"]
              if e["relation"] in ("calls", "indirect_call")
              and str(lbl.get(e["target"])).startswith(".")]
    assert leaked == []


# ── Shorthand methods inside call initializers (#2552 path) ──────────────────

def test_shorthand_method_inside_call_initializer_is_walked(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "svc.js": ("import { helper } from './payload';\n"
                   "export const Service = create({ run() { return helper(); } });\n"),
    })
    assert ("Service", "helper()") in _calls(r)


# ── Boundaries: scope, classes, dynamic dispatch ─────────────────────────────

def test_function_local_object_literal_is_not_noded(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "use.js": ("import { helper } from './payload';\n"
                   "export function f() {\n"
                   "  const local = { m: () => helper() };\n"
                   "  return local;\n"
                   "}\n"),
    })
    assert ".m()" not in _labels(r)
    assert ("f()", "helper()") in _calls(r)


def test_class_expression_binding_is_not_object_scanned(tmp_path):
    r = _extract(tmp_path, {
        "use.js": "export const Ctor = class {\n  classMethod() { return 1; }\n};\n",
    })
    assert not any(owner == "Ctor" and rel == "method"
                   for owner, rel in _owners(r, ".classMethod()"))


def test_dynamic_dispatch_through_config_stays_unresolved(tmp_path):
    r = _extract(tmp_path, {
        "payload.js": _PAYLOAD,
        "config.js": ("import { buildPayload } from './payload';\n"
                      "export const CONFIG = { bulk: { submit: (p) => buildPayload(p) } };\n"
                      "export const getConfig = (t) => CONFIG[t] || CONFIG;\n"),
        "use.js": ("import { getConfig, CONFIG } from './config';\n"
                   "export function save(t, p) {\n"
                   "  const modeConfig = getConfig(t).bulk;\n"
                   "  modeConfig.submit(p);\n"
                   "  return CONFIG.bulk.submit(p);\n"
                   "}\n"),
    })
    assert ("save()", ".submit()") not in _calls(r)


def test_non_function_property_values_add_no_members(tmp_path):
    r = _extract(tmp_path, {
        "use.js": ("export const SETTINGS = {\n"
                   "  title: 'x', limits: { max: 3 }, list: [1, 2], ref: SOMETHING.value,\n"
                   "};\n"),
    })
    assert not [m for m in _labels(r) if m.startswith(".")]
