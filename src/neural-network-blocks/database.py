
import json
from pathlib import Path

from truth_tables import generate_truth_table, get_gate_arity
from gates import get_exact_params


def make_database_entry(gate_name, implementation_entries):
    gate_name = gate_name.upper()
    return {
        "name": gate_name,
        "logic_type": "boolean",
        "description": f"{gate_name} Boolean logic gate",
        "inputs": [f"x{i+1}" for i in range(get_gate_arity(gate_name))],
        "outputs": ["y"],
        "truth_table": generate_truth_table(gate_name),
        "implementations": implementation_entries
    }


def save_database_entry(entry, output_dir="database/boolean"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / f"{entry['name']}.json"

    with open(file_path, "w") as f:
        json.dump(entry, f, indent=2)

    return file_path


def make_threshold_implementation_entry(gate_name, verification):
    params = get_exact_params(gate_name)

    if params is None:
        architecture = {
            "type": "compositional_threshold_network",
            "note": "Built from smaller verified threshold gates"
        }
        weights = None
    else:
        architecture = {
            "type": "single_threshold_neuron",
            "input_dim": len(params["weights"]),
            "output_dim": 1,
            "activation": "hard_threshold"
        }
        weights = params

    return {
        "id": f"{gate_name.lower()}_threshold_exact",
        "implementation_type": "exact_threshold",
        "architecture": architecture,
        "weights": weights,
        "verification": {
            "method": "exhaustive_truth_table_equivalence",
            "verified": verification["verified"],
            "accuracy": verification["accuracy"],
            "num_cases": verification["num_cases"]
        }
    }


def make_sigmoid_implementation_entry(gate_name, verification, sharpness=20.0):
    params = get_exact_params(gate_name)

    if params is None:
        architecture = {
            "type": "compositional_sigmoid_network",
            "note": "Built from smaller sigmoid gate approximations"
        }
        weights = None
    else:
        architecture = {
            "type": "single_sigmoid_neuron",
            "input_dim": len(params["weights"]),
            "output_dim": 1,
            "activation": "sigmoid",
            "sharpness": sharpness
        }
        weights = params

    return {
        "id": f"{gate_name.lower()}_sigmoid",
        "implementation_type": "differentiable_sigmoid",
        "architecture": architecture,
        "weights": weights,
        "verification": {
            "method": "truth_table_equivalence_after_thresholding",
            "threshold": 0.5,
            "verified": verification["verified"],
            "accuracy": verification["accuracy"],
            "num_cases": verification["num_cases"]
        }
    }


def create_program_entry(name, expression, variables, truth_table, implementation_entries):
    has_comparison = any(op in expression for op in (">", "<", ">=", "<="))
    return {
        "name": name,
        "logic_type": "comparison" if has_comparison else "boolean",
        "description": f"Symbolic program: {expression}",
        "expression": expression,
        "inputs": variables,
        "outputs": ["y"],
        "truth_table": truth_table,
        "implementations": implementation_entries
    }

def make_program_implementation_entry(expression, verification):
    return {
        "id": "compiled_threshold_program",
        "implementation_type": "compiled_neural_program",
        "architecture": {
            "type": "compiled_logic_graph",
            "source": expression,
            "gate_basis": "threshold"
        },
        "weights": None,
        "verification": {
            "method": "program_equivalence",
            "verified": verification["verified"],
            "accuracy": verification["accuracy"],
            "num_cases":verification["num_cases"]
        }
    }


def make_program_sigmoid_entry(expression, verification, sharpness=20.0):
    return {
        "id": "compiled_sigmoid_program",
        "implementation_type": "compiled_sigmoid_program",
        "architecture": {
            "type": "compiled_logic_graph",
            "source": expression,
            "gate_basis": "sigmoid",
            "sharpness": sharpness
        },
        "weights": None,
        "verification": {
            "method": "program_equivalence_after_thresholding",
            "threshold": 0.5,
            "verified": verification["verified"],
            "accuracy": verification["accuracy"],
            "num_cases": verification["num_cases"]
        }
    }


def make_program_mlp_entry(name, model, verification, variables, weights_path, hidden_dim=8, output_dim=1, pytorch_path=None):
    weights = {
        "format": "json_state_dict",
        "path": weights_path
    }
    if pytorch_path is not None:
        weights["pytorch_format"] = "pytorch_state_dict"
        weights["pytorch_path"] = pytorch_path
    return {
        "id": f"{name}_mlp_trained",
        "implementation_type": "trained_mlp",
        "architecture": {
            "type": "mlp",
            "activation": "tanh_hidden_sigmoid_output",
            "input_dim": len(variables),
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        },
        "weights": weights,
        "verification": {
            "method": "post_training_truth_table_equivalence",
            "threshold": 0.5,
            "verified": verification["verified"],
            "accuracy": verification["accuracy"],
            "num_cases": verification["num_cases"]
        }
    }

def make_mlp_implementation_entry(gate_name, model, verification, output_dim=1, hidden_dim=4):
    return {
        "id": f"{gate_name.lower()}_mlp_trained",
        "implementation_type": "trained_mlp",
        "architecture": {
            "type": "mlp",
            "activation": "tanh_hidden_sigmoid_output",
            "input_dim": get_gate_arity(gate_name),
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        },
        "weights": "stored_in_pytorch_state_dict_format_in_future_version",
        "verification": {
            "method": "post_training_truth_table_equivalence",
            "threshold": 0.5,
            "verified": verification["verified"],
            "accuracy": verification["accuracy"],
            "num_cases": verification["num_cases"]
        }
    }
