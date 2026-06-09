import ast
import re

from logic_graph import VariableNode, ConstantNode, ComparisonNode, GateNode, ArithmeticNode

# Derived logic gates usable in function-call form, e.g. XOR(a, b).
# AND/OR/NOT keep their infix keyword form and are handled separately.
CALL_GATE_ARITY = {
    "NAND": 2,
    "NOR": 2,
    "XOR": 2,
    "XNOR": 2,
    "IMPLIES": 2,
    "EQUIVALENCE": 2,
}


class ExpressionParser(ast.NodeVisitor):

    def visit_Name(self, node):
        return VariableNode(node.id)

    def visit_Constant(self, node):
        return ConstantNode(node.value)

    def visit_Compare(self, node):
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            operator = ">"
        elif isinstance(op, ast.Lt):
            operator = "<"
        elif isinstance(op, ast.GtE):
            operator = ">="
        elif isinstance(op, ast.LtE):
            operator = "<="
        elif isinstance(op, ast.Eq):
            operator = "=="
        elif isinstance(op, ast.NotEq):
            operator = "!="
        else:
            raise ValueError("Unsupported comparison")
        return ComparisonNode(operator, left, right)

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            gate = "AND"
        elif isinstance(node.op, ast.Or):
            gate = "OR"
        else:
            raise ValueError("Unsupported bool op")
        children = [self.visit(v) for v in node.values]
        return GateNode(gate, children)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return GateNode("NOT", [self.visit(node.operand)])
        if isinstance(node.op, ast.USub):
            return ArithmeticNode("-", ConstantNode(0), self.visit(node.operand))
        raise ValueError("Unsupported unary op")

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            operator = "+"
        elif isinstance(node.op, ast.Sub):
            operator = "-"
        elif isinstance(node.op, ast.Mult):
            operator = "*"
        else:
            raise ValueError("Unsupported arithmetic operator")
        return ArithmeticNode(operator, self.visit(node.left), self.visit(node.right))

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported call expression")
        name = node.func.id.upper()
        if name not in CALL_GATE_ARITY:
            raise ValueError(f"Unsupported gate function: {node.func.id}")
        if node.keywords:
            raise ValueError(f"{name} does not accept keyword arguments")
        arity = CALL_GATE_ARITY[name]
        if len(node.args) != arity:
            raise ValueError(f"{name} expects {arity} arguments, got {len(node.args)}")
        children = [self.visit(arg) for arg in node.args]
        return GateNode(name, children)


def normalize_expression(expression):
    expr = re.sub(r"\bAND\b", "and", expression)
    expr = re.sub(r"\bOR\b", "or", expr)
    expr = re.sub(r"\bNOT\b", "not", expr)
    return expr


def parse_expression(expression):
    python_expr = normalize_expression(expression)
    tree = ast.parse(python_expr, mode="eval")
    parser = ExpressionParser()
    return parser.visit(tree.body)
