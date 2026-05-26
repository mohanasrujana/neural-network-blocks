from models import ThresholdGate, SigmoidGate


EXACT_GATE_PARAMS = {
    "NOT": {
        "weights": [-1.0],
        "bias": 0.5
    },
    "AND": {
        "weights": [1.0, 1.0],
        "bias": -1.5
    },
    "OR": {
        "weights": [1.0, 1.0],
        "bias": -0.5
    },
    "NAND": {
        "weights": [-1.0, -1.0],
        "bias": 1.5
    },
    "NOR": {
        "weights": [-1.0, -1.0],
        "bias": 0.5
    }
}


def create_threshold_gate(name):
    name = name.upper()
    params = EXACT_GATE_PARAMS[name]
    return ThresholdGate(params["weights"], params["bias"])


def create_sigmoid_gate(name, sharpness=20.0):
    name = name.upper()
    params = EXACT_GATE_PARAMS[name]
    return SigmoidGate(params["weights"], params["bias"], sharpness=sharpness)


def create_composed_gate(name, implementation="threshold", sharpness=20.0):
    name = name.upper()

    if implementation == "threshold":
        and_gate = create_threshold_gate("AND")
        or_gate = create_threshold_gate("OR")
        nand_gate = create_threshold_gate("NAND")
        not_gate = create_threshold_gate("NOT")
    elif implementation == "sigmoid":
        and_gate = create_sigmoid_gate("AND", sharpness)
        or_gate = create_sigmoid_gate("OR", sharpness)
        nand_gate = create_sigmoid_gate("NAND", sharpness)
        not_gate = create_sigmoid_gate("NOT", sharpness)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


def get_exact_params(name):
    return EXACT_GATE_PARAMS.get(name.upper())
