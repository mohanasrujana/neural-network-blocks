import json
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "neural-network-blocks"
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from gates import create_threshold_gate, create_sigmoid_gate, create_composed_gate
from circuits import HalfAdder, FullAdder, MUX2, MUX4, PriorityEncoder, Comparator1Bit, Comparator2Bit
from models import MLPGate
from verify import verify_model_against_truth_table,train_mlp,extract_state_dict_as_lists,DEFAULT_MLP_BASE_SEED,DEFAULT_MLP_MAX_RETRIES
from database import make_database_entry, save_database_entry, make_threshold_implementation_entry, make_sigmoid_implementation_entry, make_mlp_implementation_entry
from truth_tables import get_gate_arity

MULTI_OUTPUT_DIMS = {
    "HALF_ADDER": 2,
    "FULL_ADDER": 2,
    "PRIORITY_ENCODER": 3,
    "COMPARATOR_1BIT": 3,
    "COMPARATOR_2BIT": 3,
}
WIDE_HIDDEN_GATES = {"PRIORITY_ENCODER", "COMPARATOR_1BIT", "COMPARATOR_2BIT"}
COMPOSITIONAL_CIRCUITS = {
    "HALF_ADDER": HalfAdder,
    "FULL_ADDER": FullAdder,
    "MUX2": MUX2,
    "MUX4": MUX4,
    "PRIORITY_ENCODER": PriorityEncoder,
    "COMPARATOR_1BIT": Comparator1Bit,
    "COMPARATOR_2BIT": Comparator2Bit,
}

def get_output_dim(gate_name):
    return MULTI_OUTPUT_DIMS.get(gate_name, 1)


def get_hidden_dim(gate_name):
    return 8 if gate_name in WIDE_HIDDEN_GATES else 4


GATES = [
    "NOT",
    "AND",
    "OR",
    "NAND",
    "NOR",
    "XOR",
    "XNOR",
    "IMPLIES",
    "EQUIVALENCE",
    "HALF_ADDER",
    "FULL_ADDER",
    "MUX2",
    "MUX4",
    "PRIORITY_ENCODER",
    "COMPARATOR_1BIT",
    "COMPARATOR_2BIT"
]


def save_mlp_weights(gate_name, model):
    weights_dir = Path("database/boolean/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / f"{gate_name.lower()}_mlp_weights.json"
    with open(weights_path, "w") as f:
        json.dump(extract_state_dict_as_lists(model), f, indent=2)
    return str(weights_path)


def generate_gate_entry(gate_name):
    implementations = []
    # 1. Exact threshold implementation
    if gate_name in {"XOR", "XNOR"}:
        threshold_model = create_composed_gate(gate_name, implementation="threshold")
    elif gate_name in COMPOSITIONAL_CIRCUITS:
        threshold_model = COMPOSITIONAL_CIRCUITS[gate_name]()
    else:
        threshold_model = create_threshold_gate(gate_name)

    if threshold_model is not None:
        threshold_verification = verify_model_against_truth_table(threshold_model, gate_name)
        implementations.append(make_threshold_implementation_entry(gate_name, threshold_verification))
    # 2. Sigmoid differentiable implementation
    if gate_name in COMPOSITIONAL_CIRCUITS:
        sigmoid_model = COMPOSITIONAL_CIRCUITS[gate_name](implementation="sigmoid", sharpness=20.0)
    else:
        sigmoid_model = create_sigmoid_gate(gate_name, sharpness=20.0)
    if sigmoid_model is not None:
        sigmoid_verification = verify_model_against_truth_table(sigmoid_model, gate_name)
        implementations.append(make_sigmoid_implementation_entry(gate_name, sigmoid_verification, sharpness=20.0))

    # 3. Trained MLP implementation
    input_dim = get_gate_arity(gate_name)
    output_dim = get_output_dim(gate_name)
    hidden_dim = get_hidden_dim(gate_name)
    mlp_model = MLPGate(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    mlp_model, mlp_verification = train_mlp(mlp_model, gate_name, epochs=3000, lr=0.05, base_seed=DEFAULT_MLP_BASE_SEED, max_retries=DEFAULT_MLP_MAX_RETRIES)
    mlp_entry = make_mlp_implementation_entry(gate_name, mlp_model, mlp_verification, output_dim=output_dim, hidden_dim=hidden_dim)

    mlp_weights_path = save_mlp_weights(gate_name, mlp_model)
    mlp_entry["weights"] = {
        "format": "json_state_dict",
        "path": mlp_weights_path
    }

    implementations.append(mlp_entry)

    entry = make_database_entry(gate_name, implementations)
    return entry


def main():
    for gate in GATES:
        print(f"Generating database entry for {gate}...")
        entry = generate_gate_entry(gate)
        path = save_database_entry(entry)
        all_verified = all(impl["verification"]["verified"] for impl in entry["implementations"])
        print(f"Saved {path}")
        print(f"All implementations verified: {all_verified}")
        print("-" * 50)


if __name__ == "__main__":
    main()
