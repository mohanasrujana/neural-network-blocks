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

    # Control flow: if/else (tiered classification, not a single threshold)
    "if_above_5": """
        if x > 7:
            tier = 2
        else:
            if x > 3:
                tier = 1
            else:
                tier = 0
        return tier == 2
        """,

    # Control flow: for loop scans membership in {0, 1, 2, 3}
    "for_match_in_range": """
        hit = 0
        for i in range(4):
            hit = hit OR (x == i)
        return hit
        """,

    # Control flow: while loop counts fixed-size decrements
    "while_countdown": """
        steps = 0
        v = x
        while v > 0:
            v = v - 2
            steps = steps + 1
        return steps == 4
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
        while total > 8:
            total = total - 4
            steps = steps + 1
        return steps == 1
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
            v = x - 3
        else:
            v = x + 2
        steps = 0
        while v > 0:
            v = v - 1
            steps = steps + 1
        return steps >= 4
        """,

    # for with if: accumulate only when loop index is below x
    "for_with_if": """
        total = 0
        for i in range(4):
            if i < x:
                total = total + 1
        return total >= 2
        """,

    # if with for: only large x executes the counting loop
    "if_with_for": """
        if x > 8:
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
        while v > 6:
            inner = 0
            w = v
            while w > 1:
                w = w - 3
                inner = inner + 1
            outer = outer + inner
            v = v - 4
        return outer == 2
        """,

    # nested if: middle tier only (strictly between 3 and 7)
    "nested_if": """
        if x > 3:
            if x > 7:
                y = 2
            else:
                y = 1
        else:
            y = 0
        return y == 1
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
        return y >= 1
        """,

    # Classic algorithms (single input x, numeric outputs)
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
        return a
        """,

    "factorial": """
        fact = 1
        n = x
        while n > 1:
            fact = fact * n
            n = n - 1
        return fact
        """,

    "triangular_sum": """
        total = 0
        n = x
        while n > 0:
            total = total + n
            n = n - 1
        return total
        """,

    "power_of_two": """
        p = 1
        n = x
        while n > 0:
            p = p * 2
            n = n - 1
        return p
        """,

    "sum_of_squares": """
        total = 0
        for i in range(6):
            if i < x:
                total = total + i * i
        return total
        """,

    # More derived gates

    "equivalence_check":
        "EQUIVALENCE(a, b)",

    "nor_gate":
        "NOR(a, b)",

    # Additional comparison operators
    "at_least_5":
        "x >= 5",

    "at_most_7":
        "x <= 7",

    # Two-variable arithmetic comparisons
    "abs_diff_small":
        "(x - y) <= 2",

    "product_exceeds_20":
        "(x * y) > 20",

    "sum_at_most_8":
        "(x + y) <= 8",

    # Multi-input Boolean patterns
    "majority_vote":
        "((a AND b) OR (a AND c)) OR (b AND c)",

    "all_three_on":
        "(a AND b) AND c",

    # Arithmetic doubling with an exact target
    "double_then_check": """
        v = x + x
        return v == 18
        """,

    "distance_from_five": """
        d = x - 5
        if d < 0:
            d = 0 - d
        return d <= 2
        """,

    # Two-variable control flow
    "pick_larger": """
        if x > y:
            pick = x
        else:
            pick = y
        return pick >= 7
        """,

    "both_above_threshold": """
        if x > 4:
            if y > 4:
                flag = 1
            else:
                flag = 0
        else:
            flag = 0
        return flag
        """,

    # for-loop accumulation with an exact sum target
    "for_sum_threshold": """
        total = 0
        for i in range(5):
            total = total + x
        return total == 30
        """,

    "for_product_scan": """
        hit = 0
        for i in range(1, 4):
            hit = hit OR (x == 2 * i)
        return hit
        """,

    # Chained while loops
    "double_countdown": """
        v = x
        while v > 6:
            v = v - 2
        while v > 0:
            v = v - 1
        return v == 0
        """,

    # Repeated subtraction (Euclidean-style reduction)
    "subtract_reduce": """
        a = x
        b = y
        while b > 0:
            if a >= b:
                a = a - b
            else:
                b = b - a
        return a <= 1
        """,

    # Staircase threshold climbing
    "staircase": """
        level = 0
        for i in range(0, 4):
            if x > level + 1:
                level = level + 1
        return level == 1
        """,

    # Countdown with a coarser step size
    "countdown": """
        steps = 0
        remaining = x
        while remaining > 1:
            remaining = remaining - 2
            steps = steps + 1
        return steps == 3
        """,

    # Two-variable accumulation until a target
    "accumulate_until": """
        total = 0
        count = 0
        while total < x:
            total = total + y + 1
            count = count + 1
        return count <= 2
        """,

    # for then if: scan indices, then branch on the scan result
    "for_then_if": """
        seen = 0
        for i in range(5):
            seen = seen OR (x == i + 2)
        return seen
        """,

    # if then for then if: count indices matching a linear pattern
    "if_for_if_chain": """
        match = 0
        for i in range(3):
            match = match OR (x == 2 * i + 1)
        return match
        """,

    # while with nested for inside the loop body
    "while_with_for": """
        v = x
        total = 0
        i = 0
        while v > 3:
            for i in range(3):
                total = total + 1
            v = v - 2
        return total == 9
        """,

}
