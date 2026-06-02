from graphviz import Digraph

from logic_graph import (
    VariableNode,
    ConstantNode,
    ComparisonNode,
    GateNode,
)


class LogicGraphVisualizer:

    def __init__(self):
        self.graph = Digraph(
            comment="Logic Graph"
        )

        self.node_counter = 0

    def next_id(self):
        self.node_counter += 1
        return f"n{self.node_counter}"

    def add_node_recursive(self, node):

        node_id = self.next_id()

        # Variable
        if isinstance(node, VariableNode):

            self.graph.node(
                node_id,
                node.name,
                shape="ellipse"
            )

            return node_id

        # Constant
        elif isinstance(node, ConstantNode):

            self.graph.node(
                node_id,
                str(node.value),
                shape="box"
            )

            return node_id

        # Comparison
        elif isinstance(node, ComparisonNode):

            self.graph.node(
                node_id,
                node.operator,
                shape="diamond"
            )

            left_id = self.add_node_recursive(
                node.left
            )

            right_id = self.add_node_recursive(
                node.right
            )

            self.graph.edge(
                node_id,
                left_id
            )

            self.graph.edge(
                node_id,
                right_id
            )

            return node_id

        # Gate
        elif isinstance(node, GateNode):

            self.graph.node(
                node_id,
                node.gate,
                shape="circle"
            )

            for child in node.children:

                child_id = (
                    self.add_node_recursive(
                        child
                    )
                )

                self.graph.edge(
                    node_id,
                    child_id
                )

            return node_id

        else:
            raise ValueError(
                f"Unsupported node type: {type(node)}"
            )

    def render(
        self,
        root,
        filename="logic_graph"
    ):

        self.add_node_recursive(root)

        self.graph.render(
            filename,
            format="png",
            cleanup=True
        )

        print(
            f"Saved graph to {filename}.png"
        )

class NeuralGraphVisualizer:

    def __init__(self):

        self.graph = Digraph(
            comment="Neural Graph"
        )

        self.counter = 0

    def next_id(self):

        self.counter += 1

        return f"n{self.counter}"

    def build(self, node):

        node_id = self.next_id()

        #
        # Inputs
        #
        if isinstance(node, VariableNode):

            self.graph.node(
                node_id,
                f"Input\n{node.name}",
                shape="box"
            )

            return node_id

        #
        # Constants
        #
        elif isinstance(node, ConstantNode):

            self.graph.node(
                node_id,
                f"Const\n{node.value}",
                shape="box"
            )

            return node_id

        #
        # Comparison Nodes
        #
        elif isinstance(node, ComparisonNode):

            label = (
                f"Threshold\n"
                f"{node.operator}"
            )

            self.graph.node(
                node_id,
                label,
                shape="ellipse"
            )

            left_id = self.build(
                node.left
            )

            right_id = self.build(
                node.right
            )

            self.graph.edge(
                left_id,
                node_id
            )

            self.graph.edge(
                right_id,
                node_id
            )

            return node_id

        #
        # Logic Gates
        #
        elif isinstance(node, GateNode):

            self.graph.node(
                node_id,
                f"{node.gate}\nGate",
                shape="circle"
            )

            for child in node.children:

                child_id = self.build(
                    child
                )

                self.graph.edge(
                    child_id,
                    node_id
                )

            return node_id

        else:

            raise ValueError(
                f"Unsupported node type: {type(node)}"
            )

    def render(
        self,
        root,
        filename="neural_graph"
    ):

        self.build(root)

        self.graph.render(
            filename,
            format="png",
            cleanup=True
        )

        print(
            f"Saved neural graph to {filename}.png"
        )