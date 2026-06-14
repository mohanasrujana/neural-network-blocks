import itertools

from evaluator import evaluate
from parser import parse_expression, parse_program


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
    elif name == "HALF_ADDER":
        a, b = x
        sum_bit = a ^ b
        carry = a & b
        return [sum_bit, carry]
    elif name == "FULL_ADDER":
        a, b, cin = x
        sum_bit = a ^ b ^ cin
        carry = (
            (a & b)
            | (a & cin)
            | (b & cin)
        )
        return [sum_bit, carry]
    elif name == "MUX2":
        d0, d1, s = x
        y = (
            (d0 and not s)
                or
                (d1 and s)
            )
        return [int(y)]
    elif name == "MUX4":
        d0, d1, d2, d3, s0, s1 = x
        select = (s1 << 1) | s0
        if select == 0:
            y = d0
        elif select == 1:
            y = d1
        elif select == 2:
            y = d2
        else:
            y = d3
        return [int(y)]
    elif name == "PRIORITY_ENCODER":
        i0, i1, i2, i3 = x
        a1 = i2 | i3
        a0 = i3 | (i1 & (1 - i2))
        valid = i0 | i1 | i2 | i3
        return [a1, a0, valid]
    elif name == "COMPARATOR_1BIT":
        a, b = x
        return [int(a > b), int(a == b), int(a < b)]
    elif name == "COMPARATOR_2BIT":
        a1, a0, b1, b0 = x
        a = (a1 << 1) | a0
        b = (b1 << 1) | b0
        return [int(a > b), int(a == b), int(a < b)]
    raise ValueError(f"Unknown gate: {name}")

def get_gate_arity(name):
    name = name.upper()
    if name == "NOT":
        return 1
    elif name == "HALF_ADDER":
        return 2
    elif name == "FULL_ADDER":
        return 3
    elif name == "MUX2":
        return 3
    elif name == "MUX4":
        return 6
    elif name == "PRIORITY_ENCODER":
        return 4
    elif name == "COMPARATOR_1BIT":
        return 2
    elif name == "COMPARATOR_2BIT":
        return 4
    return 2


def extract_variables(expression):
    """
    Extract the free input variables of a program.

    Works for plain expressions and for statement programs (assigned
    variables and loop variables are not inputs).

    Example:
        "(a AND b) OR c"

    Returns:
        ["a", "b", "c"]
    """
    _, variables = parse_program(expression)
    return variables

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

def generate_equation_truth_table(expression, variable_domains):
    graph = parse_expression(expression)
    variables = sorted(variable_domains.keys())
    domains = [variable_domains[v] for v in variables]
    truth_table = []
    for values in itertools.product(*domains):
        assignment = dict(zip(variables, values))
        output = evaluate(graph, assignment)
        truth_table.append(
            {
                "input": assignment,
                "output": output
            }
        )
    return truth_table

NUMERIC_DOMAIN = range(0, 11)
BOOLEAN_DOMAIN = range(0, 2)


def is_numeric_program(expression):
    """A program is numeric if it contains any comparison operator."""
    return any(op in expression for op in (">", "<", "==", "!="))


def program_truth_table(expression):
    """Truth table for a symbolic program (expression or statements).

    Every program is lowered to a single logic graph, so all supported
    constructs (derived gates, arithmetic, comparisons, if/for/while) are
    handled by one code path. Variables range over the Boolean domain
    {0, 1} unless the program contains a comparison, in which case they
    range over the integer domain 0..10.
    """
    graph, variables = parse_program(expression)
    domain = NUMERIC_DOMAIN if is_numeric_program(expression) else BOOLEAN_DOMAIN
    truth_table = []
    for values in itertools.product(domain, repeat=len(variables)):
        assignment = dict(zip(variables, values))
        truth_table.append({
            "input": assignment,
            "output": evaluate(graph, assignment)
        })
    return truth_table
