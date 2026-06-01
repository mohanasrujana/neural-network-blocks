import torch
import torch.nn as nn

from gates import create_composed_gate, create_sigmoid_gate, create_threshold_gate


class HalfAdder(nn.Module):

    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        if implementation == "threshold":
            self.xor_gate = create_composed_gate("XOR", implementation="threshold")
            self.and_gate = create_threshold_gate("AND")
        elif implementation == "sigmoid":
            self.xor_gate = create_composed_gate(
                "XOR", implementation="sigmoid", sharpness=sharpness
            )
            self.and_gate = create_sigmoid_gate("AND", sharpness=sharpness)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def forward(self, x):
        sum_bit = self.xor_gate(x)
        carry = self.and_gate(x)
        return torch.cat([sum_bit, carry], dim=1)