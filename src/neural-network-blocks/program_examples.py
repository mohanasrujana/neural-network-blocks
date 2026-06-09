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

}
