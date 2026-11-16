"""JS/TS instance-method calls on constructed receivers.

`_resolve_typescript_member_calls` types a member-call receiver from the file's
`new` bindings, but three shapes were dropped or mis-typed:

  * a direct constructor receiver, `new RouteApi().getRouteDetails()`, recorded
    no receiver at all, so only the `RouteApi` construction was linked;
  * a capitalized local, `let Route = new RouteApi(); Route.getRouteDetails()`,
    was taken for a class name (`Route`), found no such class, and was dropped;
  * a name bound to two different classes in one file kept whichever binding the
    walk met first, so one of the two calls was attributed to the wrong class.

A constructor named in source is EXTRACTED (like `Type.method()`); a receiver
typed through a `new` binding stays INFERRED. Conflicting bindings, unknown
receivers, builtins and types the caller cannot see emit nothing.
"""
from __future__ import annotations

from pathlib import Path

from graphify.extract import extract

_ROUTE_API = (
    "export class RouteApi {\n"
    "  async getRouteDetails(id) { return id; }\n"
    "  async getRoutesList(p) { return p; }\n"
    "  load() { return 1; }\n"
    "  static create() { return new RouteApi(); }\n"
    "}\n"
)
_ORDER_API = (
    "export class OrderApi {\n"
    "  load() { return 2; }\n"
    "}\n"
)


def _calls(tmp_path: Path, files: dict[str, str]):
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    r = extract([tmp_path / n for n in files],
                cache_root=tmp_path / "graphify-out", parallel=False)
    lbl = {n["id"]: n["label"] for n in r["nodes"]}
    return {
        (lbl.get(e["source"]), lbl.get(e["target"]), e.get("confidence"))
        for e in r["edges"] if e["relation"] == "calls"
    }, r


def _targets(calls, caller: str, method: str) -> set[str]:
    """Confidences of `caller -> .method()` call edges."""
    return {c for s, t, c in calls if s == f"{caller}()" and t == f".{method}()"}


def _method_edges(r, caller: str, method: str) -> list[dict]:
    """`caller -> X.method()` call edges with their target ids (to tell owners apart)."""
    ids = {n["id"]: n for n in r["nodes"]}
    return [e for e in r["edges"]
            if e["relation"] == "calls"
            and ids.get(e["source"], {}).get("label") == f"{caller}()"
            and ids.get(e["target"], {}).get("label") == f".{method}()"]


# ── Direct constructor receivers ─────────────────────────────────────────────

def test_direct_constructor_receiver_resolves_extracted(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function load(id) { return new RouteApi().getRouteDetails(id); }\n"),
    })
    assert _targets(calls, "load", "getRouteDetails") == {"EXTRACTED"}
    # the construction itself is still linked, as before
    assert ("load()", "RouteApi", "EXTRACTED") in calls


def test_parenthesized_constructor_receiver_resolves(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function load(id) { return (new RouteApi()).getRouteDetails(id); }\n"),
    })
    assert _targets(calls, "load", "getRouteDetails") == {"EXTRACTED"}


def test_constructor_receiver_inside_callback_and_promise_chain(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.jsx": ("import { RouteApi } from './api/route';\n"
                    "export function Panel({ id }) {\n"
                    "  useEffect(() => {\n"
                    "    new RouteApi().getRouteDetails(id).then((r) => r);\n"
                    "  }, [id]);\n"
                    "  return null;\n"
                    "}\n"),
    })
    assert _targets(calls, "Panel", "getRouteDetails") == {"EXTRACTED"}


def test_constructor_receiver_in_typescript(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.ts": _ROUTE_API,
        "use.ts": ("import { RouteApi } from './api/route';\n"
                   "export async function load(id: string) {\n"
                   "  return await new RouteApi().getRouteDetails(id);\n"
                   "}\n"),
    })
    assert _targets(calls, "load", "getRouteDetails") == {"EXTRACTED"}


# ── Variables initialized from constructors ──────────────────────────────────

def test_lowercase_const_binding_still_inferred(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function load(id) { const route = new RouteApi(); "
                   "return route.getRouteDetails(id); }\n"),
    })
    assert _targets(calls, "load", "getRouteDetails") == {"INFERRED"}


def test_capitalized_let_binding_resolves_inferred(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export async function load(id) {\n"
                   "  let Route = new RouteApi();\n"
                   "  let data = await Route.getRouteDetails(id);\n"
                   "  return data;\n"
                   "}\n"),
    })
    assert _targets(calls, "load", "getRouteDetails") == {"INFERRED"}


def test_same_name_same_type_in_two_functions_is_not_a_conflict(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function a(id) { let Route = new RouteApi(); return Route.getRouteDetails(id); }\n"
                   "export function b() { let Route = new RouteApi(); return Route.getRoutesList({}); }\n"),
    })
    assert _targets(calls, "a", "getRouteDetails") == {"INFERRED"}
    assert _targets(calls, "b", "getRoutesList") == {"INFERRED"}


def test_capitalized_class_receiver_still_static_extracted(tmp_path):
    # A real class name next to a variable binding in the same file: the class
    # stays a type reference (EXTRACTED), the variable is typed by the table.
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function make() { return RouteApi.create(); }\n"
                   "export function use(id) { let Route = new RouteApi(); "
                   "return Route.getRouteDetails(id); }\n"),
    })
    assert _targets(calls, "make", "create") == {"EXTRACTED"}
    assert _targets(calls, "use", "getRouteDetails") == {"INFERRED"}


# ── Conflicting bindings stay unresolved ─────────────────────────────────────

def test_name_bound_to_two_classes_in_one_file_emits_nothing(tmp_path):
    _, r = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "api/order.js": _ORDER_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "import { OrderApi } from './api/order';\n"
                   "export function a() { let Api = new RouteApi(); return Api.load(); }\n"
                   "export function b() { let Api = new OrderApi(); return Api.load(); }\n"),
    })
    assert _method_edges(r, "a", "load") == []
    assert _method_edges(r, "b", "load") == []


def test_reassignment_to_another_class_emits_nothing(tmp_path):
    _, r = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "api/order.js": _ORDER_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "import { OrderApi } from './api/order';\n"
                   "export function a(flag) {\n"
                   "  let api = new RouteApi();\n"
                   "  if (flag) { api = new OrderApi(); }\n"
                   "  return api.load();\n"
                   "}\n"),
    })
    assert _method_edges(r, "a", "load") == []


def test_reassignment_alone_does_not_type_a_receiver(tmp_path):
    # Lazy-init: the binding's initializer is not a constructor, so the name is
    # not typed (and the file-wide table cannot lend it to another scope).
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function a(cache, id) {\n"
                   "  let api = cache.get(id);\n"
                   "  if (!api) { api = new RouteApi(); }\n"
                   "  return api.load();\n"
                   "}\n"
                   "export function b(items) { return items.map((api) => api.load()); }\n"),
    })
    assert _targets(calls, "a", "load") == set()
    assert _targets(calls, "b", "load") == set()


def test_reassignment_to_same_class_still_resolves(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function a() { let api = new RouteApi(); api = new RouteApi(); "
                   "return api.load(); }\n"),
    })
    assert _targets(calls, "a", "load") == {"INFERRED"}


# ── Unknown receivers stay unresolved ────────────────────────────────────────

def test_untyped_receivers_emit_no_method_edge(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "export function p(x) { return x.getRouteDetails(1); }\n"
                   "export function q() { const r = makeApi(); return r.getRouteDetails(2); }\n"
                   "export function u() { return Unknown.getRouteDetails(3); }\n"),
    })
    for caller in ("p", "q", "u"):
        assert _targets(calls, caller, "getRouteDetails") == set(), caller


def test_constructor_of_unimported_class_emits_no_method_edge(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": "export function load(id) { return new RouteApi().getRouteDetails(id); }\n",
    })
    assert _targets(calls, "load", "getRouteDetails") == set()


def test_builtin_constructor_receiver_does_not_bind_to_user_method(tmp_path):
    calls, _ = _calls(tmp_path, {
        "store.js": "export class Store {\n  get(k) { return k; }\n}\n",
        "use.js": ("import { Store } from './store';\n"
                   "export function read(k) { return new Map().get(k); }\n"),
    })
    assert _targets(calls, "read", "get") == set()


def test_lowercase_constructor_call_receiver_emits_nothing(tmp_path):
    calls, _ = _calls(tmp_path, {
        "api/route.js": _ROUTE_API,
        "use.js": ("import { RouteApi } from './api/route';\n"
                   "const factory = RouteApi;\n"
                   "export function load(id) { return new factory().getRouteDetails(id); }\n"),
    })
    assert _targets(calls, "load", "getRouteDetails") == set()
