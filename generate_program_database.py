import json
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "neural-network-blocks"
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from program_examples import PROGRAMS
from truth_tables import extract_variables, program_truth_table
from database import create_program_entry, make_program_implementation_entry
from compiler import compile_program
from verify import verify_program

PROGRAM_DB_DIR = Path("database/programs")
PROGRAM_DB_DIR.mkdir(parents=True,exist_ok=True)

def generate_program_entry(name, expression):
    variables = extract_variables(expression)

    truth_table = program_truth_table(expression)
    model = compile_program(expression)
    verification = verify_program(model, expression)
    implementation_entries = [make_program_implementation_entry(expression, verification)]
    return create_program_entry(name, expression, variables, truth_table, implementation_entries)

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