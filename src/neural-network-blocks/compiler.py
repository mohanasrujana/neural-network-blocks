import torch
import torch.nn as nn
from gates import create_threshold_gate, create_sigmoid_gate, create_composed_gate
from models import ThresholdGate
from logic_graph import VariableNode, ConstantNode, GateNode, ComparisonNode, ArithmeticNode
from parser import parse_expression
from truth_tables import extract_variables

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

    def evaluate_node(self, node, assignment):

        if isinstance(node, VariableNode):
            return assignment[node.name]

        if isinstance(node, ConstantNode):
            return torch.full_like(next(iter(assignment.values())), float(node.value))

        if isinstance(node, ArithmeticNode):
            left = self.evaluate_node(node.left, assignment)
            right = self.evaluate_node(node.right, assignment)
            if node.operator == "+":
                return left + right
            if node.operator == "-":
                return left - right
            if node.operator == "*":
                return left * right
            raise NotImplementedError(node.operator)

        if isinstance(node, ComparisonNode):
            right = self.evaluate_node(node.right, assignment)
            left = self.evaluate_node(node.left, assignment)
            return self.compare(node.operator, left, right)

        if isinstance(node, GateNode):
            gate = node.gate.upper()
            child_outputs = [self.evaluate_node(child, assignment) for child in node.children]
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
    graph = parse_expression(expression)
    variables = extract_variables(expression)
    model = CompiledProgram(
        graph, variables, implementation=implementation, sharpness=sharpness
    )
    return model

def compile_comparison(node):
    if node.operator == ">":
        threshold = node.right.value
        return ThresholdGate(weights=[1.0], bias=-threshold)
    if node.operator == "<":
        threshold = node.right.value
        return ThresholdGate(weights=[-1.0], bias=threshold)
    if node.operator == ">=":
        threshold = node.right.value
        return ThresholdGate(weights=[1.0], bias=-threshold+1)
    if node.operator == "<=":
        threshold = node.right.value
        return ThresholdGate(weights=[-1.0], bias=threshold)
    if node.operator == "==":
        threshold = node.right.value
        return ThresholdGate(weights=[1.0], bias=-threshold+1)
    raise NotImplementedError()