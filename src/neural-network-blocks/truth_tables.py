import itertools
import re

from evaluator import evaluate
from parser import parse_expression

DOMAINS = {
    "x": list(range(0, 11)),
    "temperature": list(range(0, 201))
}


def generate_binary_inputs(n_inputs: int):
    return [list(x) for x in itertools.product([0, 1], repeat=n_inputs)]


def gate_function(name, x):
    """
    x is a list of binary inputs.
    returns list output, e.g. [0] or [1].
    """

    name = name.upper()

    if name == "NOT":
        return [1 - x[0]]

    if name == "AND":
        return [int(x[0] == 1 and x[1] == 1)]

    if name == "OR":
        return [int(x[0] == 1 or x[1] == 1)]

    if name == "NAND":
        return [1 - int(x[0] == 1 and x[1] == 1)]

    if name == "NOR":
        return [1 - int(x[0] == 1 or x[1] == 1)]

    if name == "XOR":
        return [int(x[0] != x[1])]

    if name == "XNOR":
        return [int(x[0] == x[1])]

    if name == "IMPLIES":
        return [int(x[0] == 0 or x[1] == 1)]

    if name == "EQUIVALENCE":
        return [int(x[0] == x[1])]

    raise ValueError(f"Unknown gate: {name}")


def get_gate_arity(name):
    name = name.upper()

    if name == "NOT":
        return 1

    return 2


def extract_variables(expression):
    """
    Extract variable names from a Boolean expression.

    Example:
        "(a AND b) OR c"

    Returns:
        ["a", "b", "c"]
    """

    keywords = {
        "AND",
        "OR",
        "NOT",
        "TRUE",
        "FALSE"
    }

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)

    variables = sorted(
        {
            token
            for token in tokens
            if token.upper() not in keywords
        }
    )

    return variables

def evaluate_expression(expression, assignment):
    """
    Evaluate a Boolean expression.

    Example:

        expression:
            "(a AND b) OR c"

        assignment:
            {
                "a": 1,
                "b": 0,
                "c": 1
            }

    Returns:
        0 or 1
    """

    python_expr = expression

    python_expr = python_expr.replace("AND", "and")
    python_expr = python_expr.replace("OR", "or")
    python_expr = python_expr.replace("NOT", "not")

    result = eval(
        python_expr,
        {},
        assignment
    )

    return int(bool(result))

def generate_truth_table(name):
    n_inputs = get_gate_arity(name)
    inputs = generate_binary_inputs(n_inputs)

    table = []
    for x in inputs:
        y = gate_function(name, x)
        table.append({
            "input": x,
            "output": y
        })

    return table

def generate_equation_truth_table(
        expression,
        variable_domains
):

    graph = parse_expression(
        expression
    )

    variables = sorted(
        variable_domains.keys()
    )

    domains = [
        variable_domains[v]
        for v in variables
    ]

    truth_table = []

    for values in itertools.product(*domains):

        assignment = dict(
            zip(
                variables,
                values
            )
        )

        output = evaluate(
            graph,
            assignment
        )

        truth_table.append(
            {
                "input": assignment,
                "output": output
            }
        )

    return truth_table

def generate_program_truth_table(expression):
    """
    Generate a complete truth table for a Boolean program.

    Example:

        "(a AND b) OR c"

    Returns:

        [
            {
                "input": {
                    "a": 0,
                    "b": 0,
                    "c": 0
                },
                "output": 0
            },
            ...
        ]
    """

    variables = extract_variables(expression)

    combinations = generate_binary_inputs(
        len(variables)
    )

    truth_table = []

    for combo in combinations:

        assignment = {
            var: value
            for var, value in zip(
                variables,
                combo
            )
        }

        output = evaluate_expression(
            expression,
            assignment
        )

        truth_table.append(
            {
                "input": assignment,
                "output": output
            }
        )

    return truth_table


# def print_truth_table(expression):
#     """
#     Pretty-print a truth table.
#     """

#     table = generate_program_truth_table(
#         expression
#     )

#     variables = extract_variables(expression)

#     header = variables + ["output"]

#     print(header)

#     for row in table:

#         values = [
#             row["input"][v]
#             for v in variables
#         ]

#         values.append(
#             row["output"]
#         )

#         print(values)