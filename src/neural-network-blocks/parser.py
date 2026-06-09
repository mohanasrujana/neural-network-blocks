import ast
import re
import textwrap

from logic_graph import VariableNode, ConstantNode, ComparisonNode, GateNode, ArithmeticNode

# Inputs range over 0..10, so a countdown-style while loop needs at most
# 11 iterations. Loops that would run longer are truncated at this bound.
WHILE_MAX_ITERATIONS = 11

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

    def __init__(self, env=None):
        # Maps assigned variable names to their current logic-graph value,
        # so statement programs substitute graphs for local variables.
        self.env = env if env is not None else {}

    def visit_Name(self, node):
        if node.id in self.env:
            return self.env[node.id]
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


def _select(condition, then_node, else_node):
    """condition*then + (1 - condition)*else. Exact for 0/1 conditions."""
    return ArithmeticNode(
        "+",
        ArithmeticNode("*", condition, then_node),
        ArithmeticNode("*", ArithmeticNode("-", ConstantNode(1), condition), else_node),
    )


class ProgramLowerer:
    """Lower a statement program (assignments, if/else, for, while, return)
    to a single logic graph by symbolic execution.

    for loops are unrolled over their constant range, while loops are
    unrolled up to WHILE_MAX_ITERATIONS with the loop condition guarding
    each iteration's updates, and if/else branches are merged with
    arithmetic selects. The result is one expression graph that the
    evaluator and the neural compiler already understand.
    """

    def lower(self, statements):
        env = {}
        for index, stmt in enumerate(statements):
            if isinstance(stmt, ast.Return):
                if index != len(statements) - 1:
                    raise ValueError("return must be the final statement")
                return self._expr(stmt.value, env)
            self._exec(stmt, env)
        raise ValueError("program must end with a return statement")

    def _expr(self, node, env):
        return ExpressionParser(env).visit(node)

    def _exec(self, stmt, env):
        if isinstance(stmt, ast.Assign):
            (target,) = stmt.targets
            if not isinstance(target, ast.Name):
                raise ValueError("only simple variable assignments are supported")
            env[target.id] = self._expr(stmt.value, env)
        elif isinstance(stmt, ast.If):
            self._exec_if(stmt, env)
        elif isinstance(stmt, ast.For):
            self._exec_for(stmt, env)
        elif isinstance(stmt, ast.While):
            self._exec_while(stmt, env)
        else:
            raise ValueError(f"Unsupported statement: {type(stmt).__name__}")

    def _exec_block(self, statements, env):
        for stmt in statements:
            self._exec(stmt, env)

    def _exec_if(self, stmt, env):
        condition = self._expr(stmt.test, env)
        then_env = dict(env)
        else_env = dict(env)
        self._exec_block(stmt.body, then_env)
        self._exec_block(stmt.orelse, else_env)
        for name in set(then_env) | set(else_env):
            then_val = then_env.get(name)
            else_val = else_env.get(name)
            if then_val is else_val:
                env[name] = then_val
            elif then_val is None or else_val is None:
                raise ValueError(
                    f"'{name}' must be assigned in both branches or before the if"
                )
            else:
                env[name] = _select(condition, then_val, else_val)

    def _exec_for(self, stmt, env):
        call = stmt.iter
        is_constant_range = (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and all(isinstance(arg, ast.Constant) for arg in call.args)
        )
        if not is_constant_range or not isinstance(stmt.target, ast.Name):
            raise ValueError("for loops must iterate over range() with constant bounds")
        for value in range(*[arg.value for arg in call.args]):
            env[stmt.target.id] = ConstantNode(value)
            self._exec_block(stmt.body, env)

    def _exec_while(self, stmt, env):
        for _ in range(WHILE_MAX_ITERATIONS):
            condition = self._expr(stmt.test, env)
            body_env = dict(env)
            self._exec_block(stmt.body, body_env)
            for name, value in body_env.items():
                if env.get(name) is not value:
                    if name not in env:
                        raise ValueError(
                            f"'{name}' must be initialized before the while loop"
                        )
                    env[name] = _select(condition, value, env[name])


def collect_variables(graph):
    """Sorted free input variables of a logic graph (DAG-safe)."""
    names = set()
    visited = set()
    stack = [graph]
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, VariableNode):
            names.add(node.name)
        elif isinstance(node, GateNode):
            stack.extend(node.children)
        elif isinstance(node, (ComparisonNode, ArithmeticNode)):
            stack.extend([node.left, node.right])
    return sorted(names)


def parse_program(source):
    """Parse a single expression or a multi-statement program.

    Statement programs use assignments, if/else, for over range(),
    bounded while loops, and end with a return statement.

    Returns (graph, variables) where variables are the sorted free
    input variables.
    """
    text = textwrap.dedent(normalize_expression(source)).strip()
    try:
        tree = ast.parse(text, mode="eval")
        graph = ExpressionParser().visit(tree.body)
    except SyntaxError:
        # Wrap in a function so 'return' parses at statement level.
        wrapped = "def __program__():\n" + textwrap.indent(text, "    ")
        statements = ast.parse(wrapped, mode="exec").body[0].body
        graph = ProgramLowerer().lower(statements)
    return graph, collect_variables(graph)
