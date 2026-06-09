class Node:
    pass


class VariableNode(Node):
    def __init__(self, name):
        self.name = name


class ConstantNode(Node):
    def __init__(self, value):
        self.value = value


class GateNode(Node):
    def __init__(self, gate, children):
        self.gate = gate
        self.children = children


class ComparisonNode(Node):
    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right


class ArithmeticNode(Node):
    """Linear arithmetic on numeric operands (e.g. x + y, 2 * x).

    Only appears on the numeric side of a comparison; it never produces a
    Boolean value on its own.
    """

    def __init__(self, operator, left, right):
        self.operator = operator
        self.left = left
        self.right = right