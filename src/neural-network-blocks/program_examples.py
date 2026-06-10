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

    # Loop combinations: nested for loops
    "nested_for_pairs": """
        count = 0
        for i in range(3):
            for j in range(3):
                if x == i + j:
                    count = count + 1
        return count >= 2
        """,

    # Loop combinations: for loop feeding a while loop
    "for_then_while": """
        total = 0
        for i in range(3):
            total = total + x
        steps = 0
        while total > 4:
            total = total - 5
            steps = steps + 1
        return steps >= 2
        """,

    # Loop combinations: while loop with branching inside
    "while_with_if": """
        v = x
        steps = 0
        while v > 0:
            if v > 5:
                v = v - 3
            else:
                v = v - 1
            steps = steps + 1
        return steps <= 4
        """,

    # while then for: while shrinks x, then for scans the result
    "while_then_for": """
        v = x
        while v > 3:
            v = v - 2
        found = 0
        for i in range(3):
            found = found OR (v == i)
        return found
        """,

    # if then while: branch picks starting value, while counts down
    "if_with_while": """
        if x > 5:
            v = x
        else:
            v = 2
        steps = 0
        while v > 0:
            v = v - 1
            steps = steps + 1
        return steps >= 3
        """,

    # for with if: accumulate only when loop index is below x
    "for_with_if": """
        total = 0
        for i in range(4):
            if i < x:
                total = total + 1
        return total >= 2
        """,

    # if with for: high x runs a counting loop, else skips it
    "if_with_for": """
        if x > 3:
            count = 0
            for i in range(3):
                count = count + 1
        else:
            count = 0
        return count >= 2
        """,

    # nested while: outer loop wraps an inner countdown
    "nested_while": """
        outer = 0
        inner = 0
        w = 0
        v = x
        while v > 4:
            inner = 0
            w = v
            while w > 0:
                w = w - 2
                inner = inner + 1
            outer = outer + inner
            v = v - 3
        return outer >= 2
        """,

    # nested if: tiered thresholds on x
    "nested_if": """
        if x > 3:
            if x > 7:
                y = 2
            else:
                y = 1
        else:
            y = 0
        return y >= 1
        """,

    # multiple if/else: elif-style ladder via nested else branches
    "multiple_ifs_else": """
        if x > 8:
            y = 3
        else:
            if x > 5:
                y = 2
            else:
                if x > 2:
                    y = 1
                else:
                    y = 0
        return y >= 2
        """,

    # Classic algorithms (single input x, boolean threshold on the result)
    "fibonacci": """
        a = 0
        b = 1
        tmp = 0
        n = x
        while n > 0:
            tmp = a + b
            a = b
            b = tmp
            n = n - 1
        return b >= 8
        """,

    "factorial": """
        fact = 1
        n = x
        while n > 1:
            fact = fact * n
            n = n - 1
        return fact >= 24
        """,

    "triangular_sum": """
        total = 0
        n = x
        while n > 0:
            total = total + n
            n = n - 1
        return total >= 10
        """,

    "power_of_two": """
        p = 1
        n = x
        while n > 0:
            p = p * 2
            n = n - 1
        return p >= 8
        """,

    "sum_of_squares": """
        total = 0
        for i in range(6):
            if i < x:
                total = total + i * i
        return total >= 14
        """,

}
