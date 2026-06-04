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

X_GAP = 220
Y_GAP = 90
RADIUS = 26
MARGIN_X = 130
TITLE_SPACE = 90
LEGEND_SPACE = 80


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
        f'<rect width="{width}" height="{height}" fill="#fafafa"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="22" '
        f'font-weight="bold" fill="#222">{_esc(title)}</text>',
        f'<text x="{width / 2}" y="58" text-anchor="middle" font-size="13" '
        f'fill="#666">{_esc(subtitle)}</text>',
    ]


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

    parts.append("</svg>")
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
    content_h = (max_size - 1) * Y_GAP
    width = MARGIN_X * 2 + (n_layers - 1) * X_GAP
    height = TITLE_SPACE + content_h + 2 * RADIUS + LEGEND_SPACE

    def x_of(layer_index):
        return MARGIN_X + layer_index * X_GAP

    def y_positions(size):
        offset = (content_h - (size - 1) * Y_GAP) / 2
        base = TITLE_SPACE + RADIUS + offset
        return [base + i * Y_GAP for i in range(size)]

    coords = [list(zip([x_of(i)] * s, y_positions(s))) for i, s in enumerate(layer_sizes)]

    parts = _svg_header(width, height, title, subtitle)

    max_w = _abs_max(dense_layers) if dense_layers else 1.0
    show_weight_labels = sum(len(l["W"]) * len(l["W"][0]) for l in dense_layers) <= 18

    for li, layer in enumerate(dense_layers):
        W = layer["W"]
        prev = coords[li]
        cur = coords[li + 1]
        in_size = len(prev)
        for k, (x2, y2) in enumerate(cur):
            for j, (x1, y1) in enumerate(prev):
                w = W[k][j]
                color = POSITIVE_COLOR if w >= 0 else NEGATIVE_COLOR
                thickness = 0.6 + 3.6 * (abs(w) / max_w)
                parts.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="{color}" stroke-width="{thickness:.2f}" stroke-opacity="0.75"/>'
                )
                if show_weight_labels:
                    t = 0.32 + (0.36 * j / (in_size - 1) if in_size > 1 else 0.18)
                    mx = x1 + (x2 - x1) * t
                    my = y1 + (y2 - y1) * t
                    parts.append(
                        f'<text x="{mx:.1f}" y="{my - 3:.1f}" text-anchor="middle" '
                        f'font-size="9" fill="{color}">{w:.2f}</text>'
                    )

    for li, (size, layer_coords) in enumerate(zip(layer_sizes, coords)):
        is_input = li == 0
        is_output = li == n_layers - 1
        if is_input:
            fill, labels = INPUT_FILL, input_labels
        elif is_output:
            fill, labels = OUTPUT_FILL, output_labels
        else:
            fill, labels = NEURON_FILL, [""] * size

        biases = None if is_input else dense_layers[li - 1]["b"]

        for k, (x, y) in enumerate(layer_coords):
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{RADIUS}" fill="{fill}" '
                f'stroke="{NEURON_STROKE}" stroke-width="1.6"/>'
            )
            label = labels[k] if k < len(labels) else ""
            if label:
                parts.append(
                    f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                    f'font-size="13" fill="#222">{_esc(label)}</text>'
                )
            if biases is not None and k < len(biases) and biases[k] is not None:
                parts.append(
                    f'<text x="{x:.1f}" y="{y + RADIUS + 14:.1f}" text-anchor="middle" '
                    f'font-size="10" fill="#555">b={biases[k]:.2f}</text>'
                )

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

    legend_y = height - LEGEND_SPACE + 28
    parts.extend(
        [
            f'<line x1="{MARGIN_X}" y1="{legend_y}" x2="{MARGIN_X + 34}" y2="{legend_y}" '
            f'stroke="{POSITIVE_COLOR}" stroke-width="4"/>',
            f'<text x="{MARGIN_X + 42}" y="{legend_y + 4}" font-size="12" fill="#555">'
            f'positive weight</text>',
            f'<line x1="{MARGIN_X + 170}" y1="{legend_y}" x2="{MARGIN_X + 204}" y2="{legend_y}" '
            f'stroke="{NEGATIVE_COLOR}" stroke-width="4"/>',
            f'<text x="{MARGIN_X + 212}" y="{legend_y + 4}" font-size="12" fill="#555">'
            f'negative weight</text>',
            f'<text x="{MARGIN_X + 360}" y="{legend_y + 4}" font-size="12" fill="#555">'
            f'thickness &#8733; |weight|</text>',
        ]
    )

    if note:
        parts.append(
            f'<text x="{width / 2}" y="{height - 16}" text-anchor="middle" '
            f'font-size="12" fill="#a33">{_esc(note)}</text>'
        )

    parts.append("</svg>")
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


def write_index(out_dir, written):
    by_entry = {}
    for path in written:
        by_entry.setdefault(path.parent.name, []).append(path)

    rows = []
    for entry in sorted(by_entry):
        cards = "".join(
            f'<figure><img src="{p.parent.name}/{p.name}" alt="{p.stem}"/>'
            f"<figcaption>{p.stem}</figcaption></figure>"
            for p in sorted(by_entry[entry])
        )
        rows.append(f"<section><h2>{entry}</h2><div class='grid'>{cards}</div></section>")

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Neural Network Blocks - Visualizations</title><style>"
        "body{font-family:Helvetica,Arial,sans-serif;margin:24px;background:#f4f4f4;color:#222}"
        "h1{margin-bottom:4px}section{margin:28px 0}"
        ".grid{display:flex;flex-wrap:wrap;gap:16px}"
        "figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:8px;padding:8px}"
        "figure img{display:block;max-width:520px;height:auto}"
        "figcaption{font-size:12px;color:#666;text-align:center;margin-top:6px}"
        "</style></head><body>"
        "<h1>Neural Network Blocks</h1>"
        "<p>Neural networks reconstructed from the verified database.</p>"
        + "".join(rows)
        + "</body></html>"
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

    all_written = []
    for target in targets:
        for json_path in _iter_json_files(target):
            if json_path.parent.name == "weights":
                continue
            print(f"Visualizing {json_path}")
            all_written.extend(visualize_file(json_path, out_dir))

    if all_written:
        index_path = write_index(out_dir, all_written)
        print(f"\nGallery: {index_path}")
    print(f"Done. {len(all_written)} diagram(s) written under '{out_dir}/'.")


if __name__ == "__main__":
    main(sys.argv)
