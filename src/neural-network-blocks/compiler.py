import torch
import torch.nn as nn
from gates import create_threshold_gate, create_sigmoid_gate, create_composed_gate
from logic_graph import VariableNode, ConstantNode, GateNode, ComparisonNode, ArithmeticNode
from parser import parse_program

COMPOSED_GATES = {"XOR", "XNOR", "EQUIVALENCE"}


class CompiledProgram(nn.Module):

    def __init__(self, graph, variables, implementation="threshold", sharpness=20.0):
        super().__init__()
        self.graph = graph
        self.variables = variables
        self.implementation = implementation
        self.sharpness = sharpness

    def make_gate(self, gate):
        """Return a neural gate in the requested basis (threshold or sigmoid)."""
        if self.implementation == "sigmoid":
            if gate in COMPOSED_GATES:
                return create_composed_gate(
                    gate, implementation="sigmoid", sharpness=self.sharpness
                )
            return create_sigmoid_gate(gate, sharpness=self.sharpness)
        if gate in COMPOSED_GATES:
            return create_composed_gate(gate)
        return create_threshold_gate(gate)

    def compare(self, operator, left, right):
        """Hard comparison for the threshold basis, soft for the sigmoid basis.

        Inputs are integer-valued, so the sigmoid variants are shifted by 0.5
        to place the decision boundary between adjacent integers. Thresholding
        the result at 0.5 reproduces the exact integer comparison semantics.
        """
        if self.implementation == "sigmoid":
            k = self.sharpness
            if operator == ">":
                return torch.sigmoid(k * (left - right - 0.5))
            if operator == "<":
                return torch.sigmoid(k * (right - left - 0.5))
            if operator == ">=":
                return torch.sigmoid(k * (left - right + 0.5))
            if operator == "<=":
                return torch.sigmoid(k * (right - left + 0.5))
            if operator == "==":
                return torch.sigmoid(k * (0.5 - (left - right).abs()))
            if operator == "!=":
                return torch.sigmoid(k * ((left - right).abs() - 0.5))
            raise NotImplementedError(operator)

        if operator == ">":
            return (left > right).float()
        if operator == "<":
            return (left < right).float()
        if operator == ">=":
            return (left >= right).float()
        if operator == "<=":
            return (left <= right).float()
        if operator == "==":
            return (left == right).float()
        if operator == "!=":
            return (left != right).float()
        raise NotImplementedError(operator)

    def evaluate_node(self, node, assignment, memo=None):
        # Memoized per forward pass: unrolled loop graphs share subgraphs
        # heavily, so each shared node is computed only once.
        if memo is None:
            memo = {}
        key = id(node)
        if key in memo:
            return memo[key]
        result = self._evaluate_node(node, assignment, memo)
        memo[key] = result
        return result

    def _evaluate_node(self, node, assignment, memo):

        if isinstance(node, VariableNode):
            return assignment[node.name]

        if isinstance(node, ConstantNode):
            return torch.full_like(next(iter(assignment.values())), float(node.value))

        if isinstance(node, ArithmeticNode):
            left = self.evaluate_node(node.left, assignment, memo)
            right = self.evaluate_node(node.right, assignment, memo)
            if node.operator == "+":
                return left + right
            if node.operator == "-":
                return left - right
            if node.operator == "*":
                return left * right
            raise NotImplementedError(node.operator)

        if isinstance(node, ComparisonNode):
            right = self.evaluate_node(node.right, assignment, memo)
            left = self.evaluate_node(node.left, assignment, memo)
            return self.compare(node.operator, left, right)

        if isinstance(node, GateNode):
            gate = node.gate.upper()
            child_outputs = [self.evaluate_node(child, assignment, memo) for child in node.children]
            model = self.make_gate(gate)
            if gate == "NOT":
                return model(child_outputs[0])
            inputs = torch.cat(child_outputs, dim=1)
            return model(inputs)

        raise ValueError(f"Unsupported node {node}")

    def forward(self, X):
        assignment = {}
        for i, variable in enumerate(self.variables):
            assignment[variable] = (X[:, i:i+1])
        return self.evaluate_node(self.graph, assignment)


def compile_program(expression, implementation="threshold", sharpness=20.0):
    graph, variables = parse_program(expression)
    model = CompiledProgram(
        graph, variables, implementation=implementation, sharpness=sharpness
    )
    return model
