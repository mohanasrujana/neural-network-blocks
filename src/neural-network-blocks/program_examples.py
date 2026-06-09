PROGRAMS = {
    # Boolean programs (AND / OR / NOT)
    "launch": "(a AND b) OR c",

    "alarm":
        "(sensor AND door) AND NOT override",

    "emergency":
        "(fire OR smoke) AND alarm",

    # Comparison programs over integer inputs
    "greater_than_5":
        "x > 5",

    "less_than_3":
        "x < 3",

    "between_4_and_9":
        "(x > 4) AND (x < 9)",

    # Derived logic gates in function-call form
    "xor_parity":
        "XOR(a, b)",

    "three_way_parity":
        "XOR(XOR(a, b), c)",

    "mutual_exclusion":
        "NAND(a, b)",

    "implication":
        "IMPLIES(a, b)",

    # Arithmetic on the numeric side of comparisons
    "sum_exceeds_10":
        "(x + y) > 10",

    "values_equal":
        "x == y",

    "values_differ":
        "x != y",

    "weighted_threshold":
        "(2 * x) >= (y + 3)",

    # Mixed: arithmetic, comparison, and a Boolean connective together
    "sum_high_or_equal":
        "((x + y) > 10) OR (x == y)",

    # Control flow: if/else (branches merged into one logic graph)
    "if_above_5": """
if x > 5:
    y = 1
else:
    y = 0
return y
""",

    # Control flow: for loop (unrolled over a constant range)
    "for_match_in_range": """
hit = 0
for i in range(3):
    hit = hit OR (x == i)
return hit
""",

    # Control flow: while loop (unrolled with the condition guarding updates)
    "while_countdown": """
steps = 0
v = x
while v > 0:
    v = v - 2
    steps = steps + 1
return steps >= 3
""",

}
