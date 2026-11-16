"""JSX tags that render an imported component under another local name.

`<Comp />` became a `calls` edge (#jsx-usage), but only by name: the tag's text
had to equal the component's label. A component rendered through a renamed
binding -- `import { Button as MyButton }`, `import Fancy from './button'`,
`import { UserCard as Card } from './ui'` -- matched no node, and member tags
(`<UI.Button />`) were never recorded at all, so the component looked unused.

The import/export facts already map every local import binding to the symbol it
names, through barrels, `default as` re-exports and `export *`. JSX tags now
resolve through those facts, exactly like `ns.fn()` and default-binding calls:
one unambiguous exported symbol, a binding the tag does not shadow, and a node
the corpus owns. Member tags resolve only through a namespace import or a
default-exported object, so `<icons.Close>` / `<props.Comp>` / `<Ctx.Provider>`
still never bind by their last segment.
"""
from __future__ import annotations

import json
from pathlib import Path

from graphify.extract import extract
from graphify.watch import _rebuild_code

_BUTTON = "export function Button() { return null; }\n"


def _write(tmp_path: Path, files: dict[str, str]) -> list[Path]:
    out = []
    for name, body in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
        out.append(p)
    return out


def _extract(tmp_path: Path, files: dict[str, str]):
    return extract(_write(tmp_path, files), cache_root=tmp_path / "graphify-out",
                   parallel=False)


def _calls(r) -> list[tuple[str, str, str]]:
    """(caller label, callee label, callee file name) for every `calls` edge."""
    nodes = {n["id"]: n for n in r["nodes"]}
    out = []
    for e in r["edges"]:
        if e["relation"] != "calls":
            continue
        src, tgt = nodes.get(e["source"], {}), nodes.get(e["target"], {})
        out.append((src.get("label"), tgt.get("label"),
                    Path(str(tgt.get("source_file", ""))).as_posix().split("/")[-1]))
    return out


def _edges_to(r, file_name: str) -> list[tuple[str, str]]:
    return [(s, t) for s, t, f in _calls(r) if f == file_name]


# ── Renamed named and default imports ────────────────────────────────────────

def test_named_alias_tag_in_function_component(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button as MyButton } from './button';\n"
                    "export function App() { return <MyButton />; }\n"),
    })
    assert ("App()", "Button()", "button.jsx") in _calls(r)


def test_named_alias_tag_in_every_caller_kind(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button as B } from './button';\n"
                    "export const Arrow = () => <B />;\n"
                    "export class Page extends React.Component {\n"
                    "  renderRow = () => <B />;\n"
                    "  render() { return <B>{this.renderRow()}</B>; }\n"
                    "}\n"
                    "export const COLUMNS = { cell: (v) => <B v={v} /> };\n"),
    })
    edges = _edges_to(r, "button.jsx")
    for caller in ("Arrow()", ".renderRow()", ".render()", ".cell()"):
        assert (caller, "Button()") in edges, caller


def test_renamed_default_import_of_default_exported_binding(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": "function Button() { return null; }\nexport default Button;\n",
        "app.jsx": ("import Fancy from './button';\n"
                    "export function App() { return <Fancy />; }\n"),
    })
    assert ("App()", "Button()", "button.jsx") in _calls(r)


def test_renamed_default_import_of_default_function_declaration(tmp_path):
    r = _extract(tmp_path, {
        "button.tsx": "export default function Button() { return null; }\n",
        "app.tsx": ("import Fancy from './button';\n"
                    "export function App() { return <Fancy />; }\n"),
    })
    assert ("App()", "Button()", "button.tsx") in _calls(r)


def test_alias_through_barrel_and_default_as_reexport(tmp_path):
    r = _extract(tmp_path, {
        "ui/Button.jsx": _BUTTON,
        "ui/UserCard.jsx": "export default function UserCard() { return null; }\n",
        "ui/index.js": ("export { Button } from './Button';\n"
                        "export { default as UserCard } from './UserCard';\n"),
        "app.jsx": ("import { Button as B, UserCard as Card } from './ui';\n"
                    "export function App() { return <Card><B /></Card>; }\n"),
    })
    calls = _calls(r)
    assert ("App()", "Button()", "Button.jsx") in calls
    assert ("App()", "UserCard()", "UserCard.jsx") in calls


def test_alias_through_star_reexport(tmp_path):
    r = _extract(tmp_path, {
        "ui/Button.jsx": _BUTTON,
        "ui/index.js": "export * from './Button';\n",
        "app.jsx": ("import { Button as B } from './ui';\n"
                    "export function App() { return <B />; }\n"),
    })
    assert ("App()", "Button()", "Button.jsx") in _calls(r)


def test_alias_to_styled_and_memo_component_values(tmp_path):
    # Same-name tags already bind to these non-function component values.
    r = _extract(tmp_path, {
        "styles.js": ("export const StyledBox = styled.div`color: red;`;\n"
                      "export const Memo = React.memo(function Inner() { return null; });\n"),
        "app.jsx": ("import { StyledBox as Box, Memo as M } from './styles';\n"
                    "export function App() { return <Box><M /></Box>; }\n"),
    })
    edges = _edges_to(r, "styles.js")
    assert ("App()", "StyledBox") in edges
    assert ("App()", "Memo") in edges


# ── Precision: no guessing ───────────────────────────────────────────────────

def test_alias_named_like_another_component_binds_only_to_the_import(tmp_path):
    r = _extract(tmp_path, {
        "other/Button.jsx": _BUTTON,
        "fancy.jsx": "export function Fancy() { return null; }\n",
        "app.jsx": ("import { Fancy as Button } from './fancy';\n"
                    "export function App() { return <Button />; }\n"),
    })
    assert _edges_to(r, "fancy.jsx") == [("App()", "Fancy()")]
    assert _edges_to(r, "Button.jsx") == []


def test_ambiguous_star_exports_resolve_nothing(tmp_path):
    r = _extract(tmp_path, {
        "ui/a.jsx": _BUTTON,
        "ui/b.jsx": _BUTTON,
        "ui/index.js": "export * from './a';\nexport * from './b';\n",
        "app.jsx": ("import { Button as B } from './ui';\n"
                    "export function App() { return <B />; }\n"),
    })
    assert _edges_to(r, "a.jsx") == []
    assert _edges_to(r, "b.jsx") == []


def test_tag_shadowed_by_a_local_binding_is_not_resolved(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button as B } from './button';\n"
                    "export function Prop({ B }) { return <B />; }\n"
                    "export function Local(props) { const B = props.as; return <B />; }\n"),
    })
    assert _edges_to(r, "button.jsx") == []


def test_package_import_alias_resolves_nothing(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button as B } from 'antd';\n"
                    "export function App() { return <B />; }\n"),
    })
    assert _edges_to(r, "button.jsx") == []


def test_module_object_bindings_are_not_components(tmp_path):
    # `import * as Kit` and a whole-module `require()` bind the module object,
    # never its default export.
    r = _extract(tmp_path, {
        "button.jsx": "export default function Button() { return null; }\n",
        "ns.jsx": ("import * as Kit from './button';\n"
                   "export function Ns() { return <Kit />; }\n"),
        "cjs.jsx": ("const Fancy = require('./button');\n"
                    "export function Cjs() { return <Fancy />; }\n"),
    })
    assert _edges_to(r, "button.jsx") == []


def test_lowercase_alias_tag_is_an_intrinsic_element(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button as button } from './button';\n"
                    "export function App() { return <button />; }\n"),
    })
    assert _edges_to(r, "button.jsx") == []


def test_alias_and_same_name_use_give_one_edge(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button, Button as B } from './button';\n"
                    "export function App() { return <div><Button /><B /><B /></div>; }\n"),
    })
    assert _edges_to(r, "button.jsx") == [("App()", "Button()")]


def test_same_name_tag_keeps_exactly_one_edge(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button } from './button';\n"
                    "export function App() { return <Button />; }\n"),
    })
    assert _edges_to(r, "button.jsx") == [("App()", "Button()")]


# ── Member tags through namespace imports and default objects ────────────────

def test_namespace_member_tag(tmp_path):
    r = _extract(tmp_path, {
        "ui/Button.jsx": _BUTTON,
        "ui/index.js": "export { Button } from './Button';\n",
        "app.jsx": ("import * as UI from './ui';\n"
                    "export function App() { return <UI.Button />; }\n"),
    })
    assert ("App()", "Button()", "Button.jsx") in _calls(r)


def test_default_object_member_tag(tmp_path):
    r = _extract(tmp_path, {
        "ui.jsx": "export default {\n  Button() { return null; },\n};\n",
        "app.jsx": ("import UI from './ui';\n"
                    "export function App() { return <UI.Button />; }\n"),
    })
    assert ("App()", "Button()", "ui.jsx") in _calls(r)


def test_member_tag_on_shadowed_namespace_is_not_resolved(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import * as UI from './button';\n"
                    "export function App({ UI }) { return <UI.Button />; }\n"),
    })
    assert _edges_to(r, "button.jsx") == []


def test_member_tag_on_non_import_receiver_binds_nothing(tmp_path):
    r = _extract(tmp_path, {
        "button.jsx": _BUTTON,
        "app.jsx": ("import { Button } from './button';\n"
                    "const Kit = { Button };\n"
                    "export function App(props) {\n"
                    "  return <div><Kit.Button /><props.Button /></div>;\n"
                    "}\n"),
    })
    assert _edges_to(r, "button.jsx") == []


def test_member_tag_is_not_resolved_by_receiver_type(tmp_path):
    # Member tags resolve through import facts only; a typed local receiver is
    # left alone rather than guessed through the member-call resolvers.
    r = _extract(tmp_path, {
        "theme.tsx": "export class Theme { Provider() { return null; } }\n",
        "app.tsx": ("import { Theme } from './theme';\n"
                    "const t: Theme = new Theme();\n"
                    "export function App() { return <t.Provider />; }\n"),
    })
    assert ("App()", ".Provider()") not in _edges_to(r, "theme.tsx")


def test_member_tag_for_a_name_the_namespace_does_not_export(tmp_path):
    r = _extract(tmp_path, {
        "icons.jsx": "export function X() { return null; }\n",
        "other.jsx": "export function Close() { return null; }\n",
        "app.jsx": ("import * as icons from './icons';\n"
                    "export function App() { return <icons.Close />; }\n"),
    })
    assert _edges_to(r, "other.jsx") == []
    assert _edges_to(r, "icons.jsx") == []


# ── Determinism and incremental updates ──────────────────────────────────────

_PROJECT = {
    "ui/Button.jsx": _BUTTON,
    "ui/Card.jsx": "export default function Card() { return null; }\n",
    "ui/index.js": ("export { Button } from './Button';\n"
                    "export { default as Card } from './Card';\n"),
    "app.jsx": ("import { Button as B, Card as C } from './ui';\n"
                "import Fancy from './ui/Card';\n"
                "import * as UI from './ui';\n"
                "export function App() { return <C><B /><Fancy /><UI.Button /></C>; }\n"),
}


def test_resolution_is_deterministic(tmp_path):
    a = _extract(tmp_path / "a", _PROJECT)
    b = _extract(tmp_path / "b", _PROJECT)
    assert sorted(_calls(a)) == sorted(_calls(b))
    assert ("App()", "Button()", "Button.jsx") in _calls(a)


def _graph_calls(root: Path) -> set[tuple[str, str]]:
    data = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    return {(link["source"], link["target"]) for link in data["links"]
            if link["relation"] == "calls"}


def _alias_targets(root: Path) -> set[tuple[str, str]]:
    data = json.loads((root / "graphify-out" / "graph.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in data["nodes"]}
    return {(nodes[link["source"]]["label"], nodes[link["target"]]["label"])
            for link in data["links"]
            if link["relation"] == "calls"
            and nodes.get(link["source"], {}).get("source_file") == "app.jsx"}


def _incremental_matches_full(tmp_path: Path, edit: str) -> None:
    _write(tmp_path, _PROJECT)
    assert _rebuild_code(tmp_path, no_cluster=True)
    full_before = _graph_calls(tmp_path)
    assert _alias_targets(tmp_path) == {("App()", "Button()"), ("App()", "Card()")}

    edited = tmp_path / edit
    edited.write_text(edited.read_text(encoding="utf-8") + "// touched\n", encoding="utf-8")
    assert _rebuild_code(tmp_path, incremental=True, no_cluster=True)
    incremental = _graph_calls(tmp_path)
    assert _alias_targets(tmp_path) == {("App()", "Button()"), ("App()", "Card()")}

    assert _rebuild_code(tmp_path, force=True, no_cluster=True)
    assert _graph_calls(tmp_path) == incremental == full_before


def test_incremental_update_of_the_importer_matches_full_rebuild(tmp_path):
    # Only app.jsx is re-extracted; every export it resolves through is unchanged.
    _incremental_matches_full(tmp_path, "app.jsx")


def test_incremental_update_of_the_component_matches_full_rebuild(tmp_path):
    _incremental_matches_full(tmp_path, "ui/Card.jsx")


def test_incremental_update_of_the_barrel_matches_full_rebuild(tmp_path):
    _incremental_matches_full(tmp_path, "ui/index.js")


def test_incremental_update_through_an_unchanged_import_export_bridge(tmp_path):
    # `import Card from './Card'; export { Card };` forwards an import without a
    # `from` on the export, so the import itself must be followed to reach the
    # unchanged file's `export default`.
    _write(tmp_path, {
        "ui/Button.jsx": _BUTTON,
        "ui/Card.jsx": "export default function Card() { return null; }\n",
        "ui/index.js": ("import { Button } from './Button';\n"
                        "import Card from './Card';\n"
                        "export { Button, Card };\n"),
        "app.jsx": ("import { Button as B, Card as C } from './ui';\n"
                    "export function App() { return <C><B /></C>; }\n"),
    })
    expected = {("App()", "Button()"), ("App()", "Card()")}
    assert _rebuild_code(tmp_path, no_cluster=True)
    assert _alias_targets(tmp_path) == expected
    app = tmp_path / "app.jsx"
    app.write_text(app.read_text(encoding="utf-8") + "// touched\n", encoding="utf-8")
    assert _rebuild_code(tmp_path, incremental=True, no_cluster=True)
    assert _alias_targets(tmp_path) == expected
