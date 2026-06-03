import torch
import torch.nn as nn
from gates import create_threshold_gate, create_composed_gate
from models import ThresholdGate
from logic_graph import VariableNode, ConstantNode, GateNode, ComparisonNode
from parser import parse_expression
from truth_tables import extract_variables

class CompiledProgram(nn.Module):

    def __init__(self, graph, variables):
        super().__init__()
        self.graph = graph
        self.variables = variables

    def evaluate_node(self, node, assignment):

        if isinstance(node, VariableNode):
            return assignment[node.name]

        if isinstance(node, ConstantNode):
            return torch.full_like(next(iter(assignment.values())), float(node.value))

        if isinstance(node, ComparisonNode):
            right = self.evaluate_node(node.right, assignment)
            left = self.evaluate_node(node.left, assignment)
            if node.operator == ">":
                return (left > right).float()
            elif node.operator == "<":
                return (left < right).float()
            elif node.operator == ">=":
                return (left >= right).float()
            elif node.operator == "<=":
                return (left <= right).float()
            elif node.operator == "==":
                return (left == right).float()

        if isinstance(node, GateNode):
            gate = node.gate.upper()
            child_outputs = [self.evaluate_node(child, assignment) for child in node.children]
            if gate == "NOT":
                model = create_threshold_gate("NOT")
                return model(child_outputs[0])
            elif gate == "AND":
                model = create_threshold_gate("AND")
                inputs = torch.cat(child_outputs, dim=1)
                return model(inputs)
            elif gate == "OR":
                model = create_threshold_gate("OR")
                inputs = torch.cat(child_outputs, dim=1)
                return model(inputs)
            elif gate == "XOR":
                model = create_composed_gate("XOR")
                inputs = torch.cat(child_outputs, dim=1)
                return model(inputs)

        raise ValueError(f"Unsupported node {node}")

    def forward(self, X):
        assignment = {}
        for i, variable in enumerate(self.variables):
            assignment[variable] = (X[:, i:i+1])
        return self.evaluate_node(self.graph, assignment)


def compile_program(expression):
    graph = parse_expression(expression)
    variables = extract_variables(expression)
    model = CompiledProgram(graph, variables)
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