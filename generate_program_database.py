import json
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "neural-network-blocks"
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from program_examples import PROGRAMS
from truth_tables import extract_variables, program_truth_table
from database import create_program_entry, make_program_implementation_entry, make_program_sigmoid_entry, make_program_mlp_entry
from compiler import compile_program
from verify import verify_program, train_program_mlp, extract_state_dict_as_lists

PROGRAM_DB_DIR = Path("database/programs")
PROGRAM_WEIGHTS_DIR = PROGRAM_DB_DIR / "weights"
PROGRAM_DB_DIR.mkdir(parents=True,exist_ok=True)

MLP_HIDDEN_DIM = 8
SIGMOID_SHARPNESS = 20.0


def save_program_mlp_weights(name, model):
    PROGRAM_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = PROGRAM_WEIGHTS_DIR / f"{name}_mlp_weights.json"
    with open(weights_path, "w") as f:
        json.dump(extract_state_dict_as_lists(model), f, indent=2)
    return str(weights_path)


def generate_program_entry(name, expression):
    variables = extract_variables(expression)
    truth_table = program_truth_table(expression)
    implementations = []

    # 1. Compiled threshold network (exact, hard gates)
    threshold_model = compile_program(expression, implementation="threshold")
    threshold_verification = verify_program(threshold_model, expression)
    implementations.append(
        make_program_implementation_entry(expression, threshold_verification)
    )

    # 2. Compiled sigmoid network (differentiable approximation)
    sigmoid_model = compile_program(
        expression, implementation="sigmoid", sharpness=SIGMOID_SHARPNESS
    )
    sigmoid_verification = verify_program(sigmoid_model, expression)
    implementations.append(
        make_program_sigmoid_entry(
            expression, sigmoid_verification, sharpness=SIGMOID_SHARPNESS
        )
    )

    # 3. Trained MLP (learned from the program truth table)
    mlp_model, mlp_verification, mlp_variables = train_program_mlp(
        name, expression, hidden_dim=MLP_HIDDEN_DIM
    )
    weights_path = save_program_mlp_weights(name, mlp_model)
    implementations.append(
        make_program_mlp_entry(
            name,
            mlp_model,
            mlp_verification,
            mlp_variables,
            weights_path,
            hidden_dim=MLP_HIDDEN_DIM,
        )
    )

    return create_program_entry(name, expression, variables, truth_table, implementations)

def main():
    for name, expression in PROGRAMS.items():
        print(f"Generating {name}")
        entry = generate_program_entry(name, expression)
        path = PROGRAM_DB_DIR / f"{name}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        print(f"Saved {path}")

    print("\nProgram database generated.")

if __name__ == "__main__":
    main()