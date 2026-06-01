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