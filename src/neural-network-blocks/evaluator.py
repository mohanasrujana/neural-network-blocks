from logic_graph import VariableNode,ConstantNode,GateNode,ComparisonNode

def evaluate(node, values):
    if isinstance(node, VariableNode):
        return values[node.name]

    if isinstance(node, ConstantNode):
        return node.value

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

    if isinstance(node, GateNode):
        children = [evaluate(child, values) for child in node.children]
        if node.gate == "AND":
            return int(all(children))
        if node.gate == "OR":
            return int(any(children))
        if node.gate == "NOT":
            return int(not children[0])
    raise ValueError("Unsupported node")