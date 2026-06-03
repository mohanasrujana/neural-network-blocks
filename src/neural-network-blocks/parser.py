import ast
import re

from logic_graph import VariableNode,ConstantNode,ComparisonNode,GateNode

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
        else:
            raise ValueError("Unsupported comparison")
        return ComparisonNode(operator,left,right)
           
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
        raise ValueError("Unsupported unary op")


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