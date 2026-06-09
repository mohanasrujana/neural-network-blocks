from logic_graph import VariableNode, ConstantNode, GateNode, ComparisonNode, ArithmeticNode

def evaluate(node, values):
    if isinstance(node, VariableNode):
        return values[node.name]

    if isinstance(node, ConstantNode):
        return node.value

    if isinstance(node, ArithmeticNode):
        left = evaluate(node.left, values)
        right = evaluate(node.right, values)
        if node.operator == "+":
            return left + right
        if node.operator == "-":
            return left - right
        if node.operator == "*":
            return left * right
        raise ValueError(f"Unsupported arithmetic operator {node.operator}")

    if isinstance(node, ComparisonNode):
        left = evaluate(node.left, values)
        right = evaluate(node.right, values)
        if node.operator == ">":
            return int(left > right)
        if node.operator == "<":
            return int(left < right)
        if node.operator == ">=":
            return int(left >= right)
        if node.operator == "<=":
            return int(left <= right)
        if node.operator == "==":
            return int(left == right)
        if node.operator == "!=":
            return int(left != right)

    if isinstance(node, GateNode):
        children = [evaluate(child, values) for child in node.children]
        gate = node.gate.upper()
        if gate == "AND":
            return int(all(children))
        if gate == "OR":
            return int(any(children))
        if gate == "NOT":
            return int(not children[0])
        if gate == "NAND":
            return int(not all(children))
        if gate == "NOR":
            return int(not any(children))
        if gate == "XOR":
            return int(children[0] != children[1])
        if gate in ("XNOR", "EQUIVALENCE"):
            return int(children[0] == children[1])
        if gate == "IMPLIES":
            return int((not children[0]) or children[1])
    raise ValueError("Unsupported node")
