"""Render the neural networks stored in the database as SVG diagrams.

Each database entry (gate or program) holds one or more neural
implementations. This module reconstructs the layer/neuron/weight
structure of every implementation and draws it as a standalone SVG:

    - neurons are circles, arranged left-to-right by layer
    - connections are coloured by sign (blue = positive, red = negative)
      and their thickness scales with the magnitude of the weight
    - biases and per-layer activations are annotated

It has no third-party dependencies so it runs anywhere Python does.
"""

import json
import re
import sys
from pathlib import Path

POSITIVE_COLOR = "#2c7fb8"
NEGATIVE_COLOR = "#d7301f"
NEURON_FILL = "#ffffff"
NEURON_STROKE = "#333333"
INPUT_FILL = "#e8f4ea"
OUTPUT_FILL = "#fdece8"
HIDDEN_FILL = "#f5f8ff"

X_GAP = 220
Y_GAP = 90
RADIUS = 26
MARGIN_X = 130
TITLE_SPACE = 90
LEGEND_SPACE = 80


def _format_weight(w):
    sign = "+" if w >= 0 else "-"
    return f"{sign}{abs(w):.2f}"


def _svg_interactive_styles():
    return """
<style type="text/css"><![CDATA[
  .edge { cursor: pointer; }
  .edge-line { transition: stroke-width .15s, stroke-opacity .15s; }
  .edge:hover .edge-line { stroke-opacity: 1; stroke-width: 5; }
  .edge .wlabel { transition: opacity .15s; }
  .edge:hover .wlabel rect { stroke-width: 2; }
  .neuron { cursor: pointer; transition: stroke-width .15s; }
  .neuron:hover { stroke-width: 3; }
  .neuron.dim { opacity: .35; }
  .edge.dim .edge-line { opacity: .2; }
  .edge.dim .wlabel { opacity: .25; }
  svg.hide-weights .wlabel { display: none; }
  svg.hide-biases .bias-label { display: none; }
]]></style>"""


def _label_xy(x1, y1, x2, y2, slot, total):
    """Place a weight label along an edge without stacking every label at the midpoint."""
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    if total <= 1:
        t = 0.5
    else:
        t = 0.28 + 0.44 * slot / (total - 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    side = 1 if slot % 2 == 0 else -1
    offset = 10 + (slot % 4) * 3
    return (
        x1 + t * dx + side * offset * px,
        y1 + t * dy + side * offset * py,
    )


def _weight_label_svg(x, y, weight, color, edge_id):
    text = _format_weight(weight)
    pill_w = max(44, len(text) * 7 + 12)
    pill_h = 18
    px = x - pill_w / 2
    py = y - pill_h / 2
    return (
        f'<g class="wlabel" data-edge="{edge_id}" pointer-events="none">'
        f'<rect x="{px:.1f}" y="{py:.1f}" width="{pill_w:.1f}" height="{pill_h}" '
        f'rx="5" fill="#ffffff" fill-opacity="0.95" stroke="{color}" stroke-width="1.2"/>'
        f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="11" '
        f'font-weight="700" fill="{color}">{text}</text></g>'
    )


def _edge_line_svg(x1, y1, x2, y2, weight, max_w, from_label, to_label, layer_idx, edge_idx):
    color = POSITIVE_COLOR if weight >= 0 else NEGATIVE_COLOR
    thickness = 1.0 + 4.5 * (abs(weight) / max_w)
    fmt = _format_weight(weight)
    safe_from = _esc(from_label)
    safe_to = _esc(to_label)
    return [
        f'<g class="edge" data-layer="{layer_idx}" data-edge="{edge_idx}" '
        f'data-weight="{weight:.4f}" data-from="{safe_from}" data-to="{safe_to}" '
        f'data-sign="{"pos" if weight >= 0 else "neg"}">',
        f'<title>{safe_from} → {safe_to}: {fmt}</title>',
        f'<line class="edge-line" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{color}" stroke-width="{thickness:.2f}" stroke-opacity="0.72"/>',
        "</g>",
    ]


def _abs_max(layers):
    biggest = 1e-9
    for layer in layers:
        for row in layer["W"]:
            for w in row:
                biggest = max(biggest, abs(w))
    return biggest


def load_state_dict(path):
    with open(path) as f:
        return json.load(f)


def dense_layers_from_state_dict(state_dict):
    """Group a flat state dict into ordered (weight, bias) linear layers."""
    modules = {}
    for key, value in state_dict.items():
        if key.endswith(".weight"):
            modules.setdefault(key[: -len(".weight")], {})["W"] = value
        elif key.endswith(".bias"):
            modules.setdefault(key[: -len(".bias")], {})["b"] = value

    def order(prefix):
        nums = re.findall(r"\d+", prefix)
        return int(nums[-1]) if nums else 0

    layers = []
    for prefix in sorted(modules, key=order):
        mod = modules[prefix]
        if "W" in mod:
            layers.append({"W": mod["W"], "b": mod.get("b")})
    return layers


def _activations_for(num_layers, architecture):
    """Best-effort activation label per computational layer."""
    activation = (architecture or {}).get("activation", "")
    if num_layers == 1:
        if "threshold" in activation:
            return ["step"]
        if "sigmoid" in activation:
            return ["sigmoid"]
        return [activation or "linear"]
    labels = ["tanh"] * num_layers
    labels[-1] = "sigmoid"
    return labels


def resolve_weights_path(rel_path, json_path):
    candidate = Path(rel_path)
    if candidate.exists():
        return candidate
    for parent in Path(json_path).resolve().parents:
        joined = parent / rel_path
        if joined.exists():
            return joined
    return candidate


def build_network(implementation, entry, json_path):
    """Return a renderable network description for one implementation.

    The returned dict has:
        input_labels  : labels for the input layer
        output_labels : labels for the final layer
        dense_layers  : list of {"W": [[...]], "b": [...] or None}
        activations   : activation label per dense layer
        note          : optional message when weights are unavailable
    """
    architecture = implementation.get("architecture", {})
    inputs = entry.get("inputs", [])
    outputs = entry.get("outputs", [])
    weights = implementation.get("weights")

    if isinstance(weights, dict) and "weights" in weights:
        row = [float(w) for w in weights["weights"]]
        bias = weights.get("bias", 0.0)
        dense_layers = [{"W": [row], "b": [float(bias)]}]
        input_labels = inputs or [f"x{i + 1}" for i in range(len(row))]
        return {
            "input_labels": input_labels,
            "output_labels": outputs or ["y"],
            "dense_layers": dense_layers,
            "activations": _activations_for(1, architecture),
            "note": None,
        }

    if isinstance(weights, dict) and weights.get("format") == "json_state_dict":
        path = resolve_weights_path(weights["path"], json_path)
        if not path.exists():
            return None
        dense_layers = dense_layers_from_state_dict(load_state_dict(path))
        if not dense_layers:
            return None
        in_dim = len(dense_layers[0]["W"][0])
        out_dim = len(dense_layers[-1]["W"])
        input_labels = inputs if len(inputs) == in_dim else [f"x{i + 1}" for i in range(in_dim)]
        output_labels = outputs if len(outputs) == out_dim else [f"y{i + 1}" for i in range(out_dim)]
        return {
            "input_labels": input_labels,
            "output_labels": output_labels,
            "dense_layers": dense_layers,
            "activations": _activations_for(len(dense_layers), architecture),
            "note": None,
        }

    block_label = entry.get("expression") or architecture.get("source")
    if not block_label:
        block_label = architecture.get("note") or architecture.get("type") or "compiled network"
    return {
        "schematic": True,
        "input_labels": inputs or ["x1", "x2"],
        "output_labels": outputs or ["y"],
        "block_label": block_label,
        "block_kind": architecture.get("type", "compiled"),
        "note": "Compositional network built from verified sub-blocks (no flat weights stored).",
    }


def _esc(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _svg_header(width, height, title, subtitle):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif">',
        _svg_interactive_styles(),
        f'<rect width="{width}" height="{height}" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="22" '
        f'font-weight="bold" fill="#222">{_esc(title)}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-size="13" '
        f'fill="#666">{_esc(subtitle)}</text>',
        '<g id="diagram">',
    ]


def _svg_footer(note, width, height, legend_y, edges=None):
    parts = [
        f'<line x1="{MARGIN_X}" y1="{legend_y}" x2="{MARGIN_X + 34}" y2="{legend_y}" '
        f'stroke="{POSITIVE_COLOR}" stroke-width="4"/>',
        f'<text x="{MARGIN_X + 42}" y="{legend_y + 4}" font-size="12" fill="#555">'
        f'positive (+)</text>',
        f'<line x1="{MARGIN_X + 150}" y1="{legend_y}" x2="{MARGIN_X + 184}" y2="{legend_y}" '
        f'stroke="{NEGATIVE_COLOR}" stroke-width="4"/>',
        f'<text x="{MARGIN_X + 192}" y="{legend_y + 4}" font-size="12" fill="#555">'
        f'negative (−)</text>',
        f'<text x="{MARGIN_X + 320}" y="{legend_y + 4}" font-size="12" fill="#555">'
        f'hover edge for details</text>',
    ]
    if note:
        parts.append(
            f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" '
            f'font-size="12" fill="#a33">{_esc(note)}</text>'
        )
    parts.append("</g></svg>")
    return parts


def schematic_to_svg(network, title, subtitle):
    input_labels = network["input_labels"]
    output_labels = network["output_labels"]
    max_size = max(len(input_labels), len(output_labels))
    content_h = (max_size - 1) * Y_GAP
    width = MARGIN_X * 2 + 2 * X_GAP
    height = TITLE_SPACE + content_h + 2 * RADIUS + LEGEND_SPACE

    in_x = MARGIN_X
    block_x = MARGIN_X + X_GAP
    out_x = MARGIN_X + 2 * X_GAP
    block_y = TITLE_SPACE + RADIUS + content_h / 2

    def y_positions(size):
        offset = (content_h - (size - 1) * Y_GAP) / 2
        base = TITLE_SPACE + RADIUS + offset
        return [base + i * Y_GAP for i in range(size)]

    in_y = y_positions(len(input_labels))
    out_y = y_positions(len(output_labels))

    parts = _svg_header(width, height, title, subtitle)

    bw, bh = 150, 70
    for y in in_y:
        parts.append(
            f'<line x1="{in_x:.1f}" y1="{y:.1f}" x2="{block_x - bw / 2:.1f}" '
            f'y2="{block_y:.1f}" stroke="#999" stroke-width="1.6"/>'
        )
    for y in out_y:
        parts.append(
            f'<line x1="{block_x + bw / 2:.1f}" y1="{block_y:.1f}" x2="{out_x:.1f}" '
            f'y2="{y:.1f}" stroke="#999" stroke-width="1.6"/>'
        )

    parts.append(
        f'<rect x="{block_x - bw / 2:.1f}" y="{block_y - bh / 2:.1f}" width="{bw}" '
        f'height="{bh}" rx="10" fill="#eef3fb" stroke="#2c7fb8" stroke-width="1.8"/>'
    )
    parts.append(
        f'<text x="{block_x:.1f}" y="{block_y - 6:.1f}" text-anchor="middle" '
        f'font-size="12" fill="#2c7fb8" font-weight="bold">logic network</text>'
    )
    parts.append(
        f'<text x="{block_x:.1f}" y="{block_y + 14:.1f}" text-anchor="middle" '
        f'font-size="11" fill="#444">{_esc(network["block_label"])}</text>'
    )

    for label, x, y in (
        [(l, in_x, in_y[i]) for i, l in enumerate(input_labels)]
    ):
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{RADIUS}" fill="{INPUT_FILL}" '
            f'stroke="{NEURON_STROKE}" stroke-width="1.6"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="13" '
            f'fill="#222">{_esc(label)}</text>'
        )
    for label, x, y in (
        [(l, out_x, out_y[i]) for i, l in enumerate(output_labels)]
    ):
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{RADIUS}" fill="{OUTPUT_FILL}" '
            f'stroke="{NEURON_STROKE}" stroke-width="1.6"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="13" '
            f'fill="#222">{_esc(label)}</text>'
        )

    for name, x in (("input", in_x), ("output", out_x)):
        parts.append(
            f'<text x="{x:.1f}" y="{TITLE_SPACE - 8:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#888">{_esc(name)}</text>'
        )

    if network.get("note"):
        parts.append(
            f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" '
            f'font-size="12" fill="#a33">{_esc(network["note"])}</text>'
        )

    parts.append("</g></svg>")
    return "\n".join(parts)


def network_to_svg(network, title, subtitle):
    if network.get("schematic"):
        return schematic_to_svg(network, title, subtitle)
    input_labels = network["input_labels"]
    dense_layers = network["dense_layers"]
    activations = network["activations"]
    output_labels = network["output_labels"]
    note = network["note"]

    layer_sizes = [len(input_labels)] + [len(layer["W"]) for layer in dense_layers]
    n_layers = len(layer_sizes)
    max_size = max(layer_sizes)
    y_gap = Y_GAP + max(0, max_size - 4) * 10
    content_h = (max_size - 1) * y_gap
    width = MARGIN_X * 2 + (n_layers - 1) * X_GAP
    height = TITLE_SPACE + content_h + 2 * RADIUS + LEGEND_SPACE

    def x_of(layer_index):
        return MARGIN_X + layer_index * X_GAP

    def y_positions(size):
        offset = (content_h - (size - 1) * y_gap) / 2
        base = TITLE_SPACE + RADIUS + offset
        return [base + i * y_gap for i in range(size)]

    coords = [list(zip([x_of(i)] * s, y_positions(s))) for i, s in enumerate(layer_sizes)]

    max_w = _abs_max(dense_layers) if dense_layers else 1.0
    edge_idx = 0
    label_specs = []

    neuron_labels = [list(input_labels)]
    for li, layer in enumerate(dense_layers):
        if li < len(dense_layers) - 1:
            neuron_labels.append([f"h{i + 1}" for i in range(len(layer["W"]))])
        else:
            neuron_labels.append(list(output_labels))

    parts = _svg_header(width, height, title, subtitle)

    for li, layer in enumerate(dense_layers):
        W = layer["W"]
        prev = coords[li]
        cur = coords[li + 1]
        from_names = neuron_labels[li]
        to_names = neuron_labels[li + 1]
        n_edges = len(prev) * len(cur)
        slot = 0
        for k, (x2, y2) in enumerate(cur):
            for j, (x1, y1) in enumerate(prev):
                w = W[k][j]
                parts.extend(
                    _edge_line_svg(
                        x1, y1, x2, y2, w, max_w,
                        from_names[j], to_names[k], li, edge_idx,
                    )
                )
                lx, ly = _label_xy(x1, y1, x2, y2, slot, n_edges)
                color = POSITIVE_COLOR if w >= 0 else NEGATIVE_COLOR
                label_specs.append((edge_idx, lx, ly, w, color))
                edge_idx += 1
                slot += 1

    for li, (size, layer_coords) in enumerate(zip(layer_sizes, coords)):
        is_input = li == 0
        is_output = li == n_layers - 1
        if is_input:
            fill, labels = INPUT_FILL, input_labels
        elif is_output:
            fill, labels = OUTPUT_FILL, output_labels
        else:
            fill, labels = HIDDEN_FILL, neuron_labels[li]

        biases = None if is_input else dense_layers[li - 1]["b"]

        for k, (x, y) in enumerate(layer_coords):
            nid = _esc(labels[k] if k < len(labels) else f"n{k}")
            parts.append(
                f'<g class="neuron" data-neuron="{nid}">'
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{RADIUS}" fill="{fill}" '
                f'stroke="{NEURON_STROKE}" stroke-width="1.6"/>'
            )
            label = labels[k] if k < len(labels) else ""
            if label:
                parts.append(
                    f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                    f'font-size="13" font-weight="600" fill="#222">{_esc(label)}</text>'
                )
            if biases is not None and k < len(biases) and biases[k] is not None:
                parts.append(
                    f'<text class="bias-label" x="{x:.1f}" y="{y + RADIUS + 14:.1f}" '
                    f'text-anchor="middle" font-size="10" fill="#555">'
                    f'bias {_format_weight(biases[k])}</text>'
                )
            parts.append("</g>")

        if is_input:
            layer_name = "input"
        elif is_output:
            layer_name = f"output ({activations[li - 1]})"
        else:
            layer_name = f"hidden ({activations[li - 1]})"
        top_y = TITLE_SPACE - 8
        parts.append(
            f'<text x="{x_of(li):.1f}" y="{top_y:.1f}" text-anchor="middle" '
            f'font-size="12" fill="#888">{_esc(layer_name)}</text>'
        )

    parts.append('<g class="label-layer">')
    for edge_id, lx, ly, w, color in label_specs:
        parts.append(_weight_label_svg(lx, ly, w, color, edge_id))
    parts.append("</g>")

    legend_y = height - LEGEND_SPACE + 28
    parts.extend(_svg_footer(note, width, height, legend_y))
    return "\n".join(parts)


def visualize_file(json_path, out_dir):
    json_path = Path(json_path)
    with open(json_path) as f:
        entry = json.load(f)

    name = entry.get("name", json_path.stem)
    target = Path(out_dir) / name
    target.mkdir(parents=True, exist_ok=True)

    written = []
    for impl in entry.get("implementations", []):
        network = build_network(impl, entry, json_path)
        if network is None:
            continue
        impl_id = impl.get("id", impl.get("implementation_type", "impl"))
        verification = impl.get("verification", {})
        acc = verification.get("accuracy")
        acc_text = f"accuracy {acc:.2f}" if isinstance(acc, (int, float)) else "unverified"
        subtitle = f'{impl.get("implementation_type", "")}  |  {acc_text}'
        if entry.get("expression"):
            subtitle = f'{entry["expression"]}  |  {subtitle}'
        svg = network_to_svg(network, f"{name} - {impl_id}", subtitle)
        out_path = target / f"{impl_id}.svg"
        out_path.write_text(svg)
        written.append(out_path)
        print(f"  wrote {out_path}")
    return written


def category_label(target):
    text = str(target).lower()
    if "boolean" in text:
        return "Boolean gates"
    if "program" in text:
        return "Programs"
    stem = Path(target).stem
    return stem.replace("_", " ").title() or "Networks"


def impl_badge(stem):
    s = stem.lower()
    if "mlp" in s:
        return "MLP", "mlp"
    if "sigmoid" in s:
        return "Sigmoid", "sigmoid"
    if "threshold" in s:
        return "Threshold", "threshold"
    return "Compiled", "compiled"


INDEX_STYLE = """
:root{
  --bg:#0f1320; --panel:#161b2e; --card:#1d2336; --border:#2a3350;
  --text:#e8ecf5; --muted:#9aa6c4; --accent:#5b8cff;
  --pos:#2c7fb8; --neg:#d7301f;
  --mlp:#6f7bf7; --sigmoid:#2bb3a3; --threshold:#e0813a; --compiled:#b65cd6;
}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.5}
header{padding:40px 32px 28px;background:var(--panel);border-bottom:1px solid var(--border)}
header h1{margin:0;font-size:30px;letter-spacing:-.02em}
header p{margin:8px 0 0;color:var(--muted);max-width:760px}
.stats{margin-top:18px;display:flex;gap:10px;flex-wrap:wrap}
.stat{background:var(--card);border:1px solid var(--border);border-radius:999px;
  padding:6px 14px;font-size:13px;color:var(--muted)}
.stat b{color:var(--text)}
.toolbar{position:sticky;top:0;z-index:5;display:flex;gap:14px;align-items:center;flex-wrap:wrap;
  padding:16px 32px;background:rgba(15,19,32,.92);backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
#search{flex:1;min-width:200px;max-width:420px;padding:10px 14px;border-radius:10px;
  border:1px solid var(--border);background:var(--card);color:var(--text);font-size:14px;outline:none}
#search:focus{border-color:var(--accent)}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{cursor:pointer;user-select:none;padding:7px 13px;border-radius:999px;font-size:13px;
  border:1px solid var(--border);background:var(--card);color:var(--muted);transition:.15s}
.chip:hover{color:var(--text)}
.chip.active{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:8px 32px 64px}
section{margin:34px 0 0}
section>h2{font-size:15px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  border-bottom:1px solid var(--border);padding-bottom:8px;margin:0 0 4px}
.entry{margin-top:22px;border:1px solid var(--border);border-radius:14px;overflow:hidden;background:var(--card)}
.entry-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px;
  cursor:pointer;user-select:none}
.entry-head:hover{background:rgba(91,140,255,.06)}
.entry-head h3{margin:0;font-size:17px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.entry-head .count{font-size:12px;color:var(--muted)}
.entry-head .chev{color:var(--muted);transition:transform .2s}
.entry.collapsed .chev{transform:rotate(-90deg)}
.entry.collapsed .grid{display:none}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;padding:0 18px 18px}
.card{background:#0b0e18;border:1px solid var(--border);border-radius:12px;overflow:hidden;
  transition:transform .15s,border-color .15s,box-shadow .15s;display:flex;flex-direction:column;cursor:pointer}
.card:hover{transform:translateY(-2px);border-color:var(--accent);box-shadow:0 10px 28px rgba(0,0,0,.35)}
.card .preview{display:block;background:#fafafa;position:relative;overflow:hidden}
.card img{display:block;width:100%;height:auto;pointer-events:none}
.card .meta{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:10px 12px}
.card .name{font-size:12px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.badge{font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;color:#fff;flex:none}
.badge.mlp{background:var(--mlp)} .badge.sigmoid{background:var(--sigmoid)}
.badge.threshold{background:var(--threshold)} .badge.compiled{background:var(--compiled)}
.empty{display:none;color:var(--muted);padding:40px 0;text-align:center}
#modal{position:fixed;inset:0;z-index:100;display:none;align-items:center;justify-content:center;padding:20px}
#modal.open{display:flex}
#modal .backdrop{position:absolute;inset:0;background:rgba(0,0,0,.72);backdrop-filter:blur(4px)}
#modal .panel{position:relative;z-index:1;width:min(1200px,96vw);height:min(92vh,900px);
  background:var(--panel);border:1px solid var(--border);border-radius:16px;display:flex;flex-direction:column;overflow:hidden;
  box-shadow:0 24px 80px rgba(0,0,0,.55)}
.modal-head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px;
  border-bottom:1px solid var(--border);flex-wrap:wrap}
.modal-head h2{margin:0;font-size:16px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.modal-tools{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.modal-tools button,.modal-tools label.btn{padding:7px 12px;border-radius:8px;border:1px solid var(--border);
  background:var(--card);color:var(--text);font-size:12px;cursor:pointer}
.modal-tools button:hover,.modal-tools label.btn:hover{border-color:var(--accent)}
.modal-tools label.btn{display:inline-flex;align-items:center;gap:6px}
.modal-tools input[type=checkbox]{accent-color:var(--accent)}
.modal-body{flex:1;display:grid;grid-template-columns:1fr 260px;min-height:0}
.viewer-wrap{overflow:auto;background:#fafafa;position:relative}
.viewer-wrap svg{display:block;margin:0 auto;transform-origin:top center;transition:transform .12s ease}
.detail-pane{border-left:1px solid var(--border);padding:16px;overflow:auto;background:var(--card);font-size:13px}
.detail-pane h3{margin:0 0 10px;font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.detail-pane .hint{color:var(--muted);font-size:12px;line-height:1.5}
.detail-pane .conn{margin-top:14px;padding:12px;border-radius:10px;background:var(--panel);border:1px solid var(--border)}
.detail-pane .conn .route{font-family:ui-monospace,Menlo,monospace;font-size:13px;margin-bottom:8px}
.detail-pane .conn .wval{font-size:28px;font-weight:700;line-height:1}
.detail-pane .conn .wval.pos{color:var(--pos)} .detail-pane .conn .wval.neg{color:var(--neg)}
.detail-pane .conn .meta2{margin-top:8px;color:var(--muted);font-size:12px}
.detail-pane .layer-list{margin:12px 0 0;padding:0;list-style:none;max-height:240px;overflow:auto}
.detail-pane .layer-list li{padding:6px 0;border-bottom:1px solid var(--border);font-size:11px;
  font-family:ui-monospace,Menlo,monospace;display:flex;justify-content:space-between;gap:8px}
.detail-pane .layer-list li span.w{font-weight:700}
.detail-pane .layer-list li span.w.pos{color:var(--pos)} .detail-pane .layer-list li span.w.neg{color:var(--neg)}
#closeModal{background:transparent;border:none;color:var(--muted);font-size:22px;cursor:pointer;padding:4px 8px}
#closeModal:hover{color:var(--text)}
@media(max-width:900px){.modal-body{grid-template-columns:1fr}.detail-pane{border-left:none;border-top:1px solid var(--border)}}
"""

INDEX_SCRIPT = """
const search=document.getElementById('search');
const chips=[...document.querySelectorAll('.chip')];
const cards=[...document.querySelectorAll('.card')];
const modal=document.getElementById('modal');
const viewer=document.getElementById('viewer');
const viewerWrap=document.getElementById('viewerWrap');
const modalTitle=document.getElementById('modalTitle');
const connPanel=document.getElementById('connPanel');
const connRoute=document.getElementById('connRoute');
const connWeight=document.getElementById('connWeight');
const connMeta=document.getElementById('connMeta');
const edgeList=document.getElementById('edgeList');
const showWeights=document.getElementById('showWeights');
const showBiases=document.getElementById('showBiases');
let activeType='all', zoom=1;

function apply(){
  const q=search.value.trim().toLowerCase();
  cards.forEach(c=>{
    const okType=activeType==='all'||c.dataset.type===activeType;
    const okText=!q||c.dataset.search.includes(q);
    c.style.display=okType&&okText?'':'none';
  });
  document.querySelectorAll('.entry').forEach(e=>{
    const any=[...e.querySelectorAll('.card')].some(c=>c.style.display!=='none');
    e.style.display=any?'':'none';
  });
  document.querySelectorAll('section').forEach(s=>{
    const any=[...s.querySelectorAll('.card')].some(c=>c.style.display!=='none');
    s.style.display=any?'':'none';
  });
  document.querySelector('.empty').style.display=cards.some(c=>c.style.display!=='none')?'none':'block';
}
search.addEventListener('input',apply);
chips.forEach(ch=>ch.addEventListener('click',()=>{
  chips.forEach(c=>c.classList.remove('active'));
  ch.classList.add('active');activeType=ch.dataset.type;apply();
}));

document.querySelectorAll('.entry-head').forEach(head=>{
  head.addEventListener('click',()=>head.parentElement.classList.toggle('collapsed'));
});

function fmtWeight(v){const n=Number(v);const s=n>=0?'+':'-';return s+Math.abs(n).toFixed(2);}

function clearHighlight(svg){
  svg.querySelectorAll('.edge').forEach(e=>e.classList.remove('dim'));
  svg.querySelectorAll('.neuron').forEach(n=>n.classList.remove('dim'));
  svg.querySelectorAll('.wlabel').forEach(w=>w.classList.remove('active'));
}

function highlightEdge(svg, edge){
  clearHighlight(svg);
  svg.querySelectorAll('.edge').forEach(e=>{if(e!==edge)e.classList.add('dim');});
  const from=edge.dataset.from, to=edge.dataset.to;
  svg.querySelectorAll('.neuron').forEach(n=>{
    const id=n.dataset.neuron;
    if(id!==from && id!==to) n.classList.add('dim');
  });
  const lid=edge.dataset.edge;
  if(lid!==undefined){
    svg.querySelectorAll('.wlabel').forEach(w=>{
      w.classList.toggle('active', w.dataset.edge===lid);
    });
  }
}

function bindSvg(svg){
  svg.classList.toggle('hide-weights', !showWeights.checked);
  svg.classList.toggle('hide-biases', !showBiases.checked);
  const edges=[...svg.querySelectorAll('.edge')];
  edgeList.innerHTML='';
  edges.forEach(edge=>{
    const li=document.createElement('li');
    const w=Number(edge.dataset.weight);
    const cls=w>=0?'pos':'neg';
    li.innerHTML=`<span>${edge.dataset.from} → ${edge.dataset.to}</span><span class="w ${cls}">${fmtWeight(w)}</span>`;
    li.addEventListener('mouseenter',()=>{
      highlightEdge(svg, edge);
      showConn(edge);
    });
    li.addEventListener('mouseleave',()=>{
      clearHighlight(svg);
      hideConn();
    });
    edgeList.appendChild(li);
    edge.addEventListener('mouseenter',()=>{highlightEdge(svg,edge);showConn(edge);});
    edge.addEventListener('mouseleave',()=>{clearHighlight(svg);hideConn();});
  });
  if(!edges.length){
    edgeList.innerHTML='<li class="hint">Schematic network — no flat weights stored.</li>';
  }
}

function setZoom(z){
  zoom=z;
  const svg=viewer.querySelector('svg');
  if(svg) svg.style.transform='scale('+zoom+')';
}

function showConn(edge){
  const w=Number(edge.dataset.weight);
  connRoute.textContent=edge.dataset.from+' → '+edge.dataset.to;
  connWeight.textContent=fmtWeight(w);
  connWeight.className='wval '+(w>=0?'pos':'neg');
  connMeta.textContent='Layer '+edge.dataset.layer+' · |w| = '+Math.abs(w).toFixed(4);
}
function hideConn(){
  connRoute.textContent='Hover a connection';
  connWeight.textContent='—';
  connWeight.className='wval';
  connMeta.textContent='Edge thickness scales with |weight|. Blue = positive, red = negative.';
}

async function openModal(src, title){
  modalTitle.textContent=title;
  viewer.innerHTML='Loading…';
  modal.classList.add('open');
  setZoom(1);
  hideConn();
  try{
    const res=await fetch(src);
    const text=await res.text();
    viewer.innerHTML=text;
    const svg=viewer.querySelector('svg');
    if(svg) bindSvg(svg);
  }catch(e){viewer.textContent='Failed to load diagram.';}
}
function closeModal(){modal.classList.remove('open');viewer.innerHTML='';}

cards.forEach(card=>{
  card.addEventListener('click',()=>openModal(card.dataset.src, card.dataset.title));
});
document.getElementById('closeModal').addEventListener('click', closeModal);
modal.querySelector('.backdrop').addEventListener('click', closeModal);
document.addEventListener('keydown', e=>{
  if(e.key==='Escape') closeModal();
  if(!modal.classList.contains('open')) return;
  if(e.key==='+'||e.key=='='){setZoom(Math.min(3,zoom+.15));}
  if(e.key==='-'){setZoom(Math.max(.4,zoom-.15));}
});
document.getElementById('zoomIn').addEventListener('click',()=>setZoom(Math.min(3,zoom+.15)));
document.getElementById('zoomOut').addEventListener('click',()=>setZoom(Math.max(.4,zoom-.15)));
document.getElementById('zoomReset').addEventListener('click',()=>setZoom(1));
function refreshToggles(){
  const svg=viewer.querySelector('svg');
  if(svg) bindSvg(svg);
}
showWeights.addEventListener('change', refreshToggles);
showBiases.addEventListener('change', refreshToggles);
"""


def write_index(out_dir, sections):
    total = sum(len(paths) for paths in sections.values())
    type_counts = {"mlp": 0, "sigmoid": 0, "threshold": 0, "compiled": 0}

    section_html = []
    for label in sections:
        paths = sections[label]
        by_entry = {}
        for p in paths:
            by_entry.setdefault(p.parent.name, []).append(p)

        entry_blocks = []
        for entry in sorted(by_entry):
            cards = []
            for p in sorted(by_entry[entry]):
                badge_text, badge_cls = impl_badge(p.stem)
                type_counts[badge_cls] = type_counts.get(badge_cls, 0) + 1
                src = f"{p.parent.name}/{p.name}"
                search_key = _esc(f"{entry} {p.stem} {badge_text}".lower())
                cards.append(
                    f'<div class="card" data-type="{badge_cls}" data-search="{search_key}" '
                    f'data-src="{src}" data-title="{_esc(entry)} — {_esc(p.stem)}" role="button" tabindex="0">'
                    f'<div class="preview"><img src="{src}" alt="{_esc(p.stem)}" loading="lazy"/></div>'
                    f'<div class="meta"><span class="name">{_esc(p.stem)}</span>'
                    f'<span class="badge {badge_cls}">{badge_text}</span></div></div>'
                )
            entry_blocks.append(
                f'<div class="entry"><div class="entry-head">'
                f'<h3>{_esc(entry)}</h3>'
                f'<span class="count">{len(by_entry[entry])} diagrams</span>'
                f'<span class="chev">▼</span></div>'
                f'<div class="grid">{"".join(cards)}</div></div>'
            )
        section_html.append(
            f'<section><h2>{_esc(label)} ({len(paths)})</h2>{"".join(entry_blocks)}</section>'
        )

    stats = (
        f'<span class="stat"><b>{total}</b> diagrams</span>'
        f'<span class="stat"><b>{len(sections)}</b> categories</span>'
        f'<span class="stat">MLP <b>{type_counts["mlp"]}</b></span>'
        f'<span class="stat">Sigmoid <b>{type_counts["sigmoid"]}</b></span>'
        f'<span class="stat">Threshold <b>{type_counts["threshold"]}</b></span>'
    )

    chips = (
        '<div class="chips">'
        '<span class="chip active" data-type="all">All</span>'
        '<span class="chip" data-type="threshold">Threshold</span>'
        '<span class="chip" data-type="sigmoid">Sigmoid</span>'
        '<span class="chip" data-type="mlp">MLP</span>'
        '<span class="chip" data-type="compiled">Compiled</span>'
        "</div>"
    )

    modal = (
        '<div id="modal"><div class="backdrop"></div><div class="panel">'
        '<div class="modal-head">'
        '<h2 id="modalTitle">Network</h2>'
        '<div class="modal-tools">'
        '<button id="zoomOut" type="button">−</button>'
        '<button id="zoomReset" type="button">100%</button>'
        '<button id="zoomIn" type="button">+</button>'
        '<label class="btn"><input type="checkbox" id="showWeights" checked/> Weights</label>'
        '<label class="btn"><input type="checkbox" id="showBiases" checked/> Biases</label>'
        '<button id="closeModal" type="button" aria-label="Close">×</button>'
        '</div></div>'
        '<div class="modal-body">'
        '<div class="viewer-wrap" id="viewerWrap"><div id="viewer"></div></div>'
        '<div class="detail-pane">'
        '<h3>Selected connection</h3>'
        '<div id="connPanel" class="conn">'
        '<div class="route" id="connRoute">Hover a connection</div>'
        '<div class="wval" id="connWeight">—</div>'
        '<div class="meta2" id="connMeta">Edge thickness scales with |weight|. '
        'Blue = positive, red = negative.</div></div>'
        '<h3 style="margin-top:18px">All weights</h3>'
        '<ul class="layer-list" id="edgeList"></ul>'
        '</div></div></div></div>'
    )

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Neural Network Blocks - Visualizations</title>"
        f"<style>{INDEX_STYLE}</style></head><body>"
        "<header><h1>Neural Network Blocks</h1>"
        "<p>Interactive gallery of verified neural implementations. Click any diagram to "
        "inspect weights: hover connections to highlight paths, toggle weight labels, and "
        "zoom with +/− keys or the toolbar.</p>"
        f"<div class='stats'>{stats}</div></header>"
        f"<div class='toolbar'><input id='search' type='search' "
        f"placeholder='Search gates, programs, implementations...'/>{chips}</div>"
        f"<main>{''.join(section_html)}"
        "<div class='empty'>No diagrams match your filters.</div></main>"
        f"{modal}<script>{INDEX_SCRIPT}</script></body></html>"
    )

    index_path = Path(out_dir) / "index.html"
    index_path.write_text(html)
    return index_path


def _iter_json_files(path):
    path = Path(path)
    if path.is_dir():
        yield from sorted(path.glob("*.json"))
    elif path.suffix == ".json":
        yield path


def main(argv):
    out_dir = "visualizations"
    if len(argv) > 1:
        targets = argv[1:]
    else:
        targets = ["database/boolean", "database/programs"]

    sections = {}
    all_written = []
    for target in targets:
        label = category_label(target)
        for json_path in _iter_json_files(target):
            if json_path.parent.name == "weights":
                continue
            print(f"Visualizing {json_path}")
            written = visualize_file(json_path, out_dir)
            sections.setdefault(label, []).extend(written)
            all_written.extend(written)

    if all_written:
        index_path = write_index(out_dir, sections)
        print(f"\nGallery: {index_path}")
    print(f"Done. {len(all_written)} diagram(s) written under '{out_dir}/'.")


if __name__ == "__main__":
    main(sys.argv)
