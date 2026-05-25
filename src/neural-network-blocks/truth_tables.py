import itertools


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

    raise ValueError(f"Unknown gate: {name}")


def get_gate_arity(name):
    name = name.upper()

    if name == "NOT":
        return 1

    return 2


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