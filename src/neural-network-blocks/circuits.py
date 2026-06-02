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

class FullAdder(nn.Module):
    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        if implementation == "threshold":
            xor_impl = "threshold"
            gate = create_threshold_gate
        elif implementation == "sigmoid":
            xor_impl = "sigmoid"
            gate = lambda name: create_sigmoid_gate(name, sharpness=sharpness)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        self.half_adder_1_xor = create_composed_gate(
            "XOR", implementation=xor_impl, sharpness=sharpness
        )
        self.half_adder_1_and = gate("AND")
        self.half_adder_2_xor = create_composed_gate(
            "XOR", implementation=xor_impl, sharpness=sharpness
        )
        self.half_adder_2_and = gate("AND")
        self.or_gate = gate("OR")

    def forward(self, x):
        a = x[:, 0:1]
        b = x[:, 1:2]
        cin = x[:, 2:3]

        # Half Adder 1
        sum1 = self.half_adder_1_xor(torch.cat([a, b], dim=1))
        carry1 = self.half_adder_1_and(torch.cat([a, b], dim=1))

        # Half Adder 2
        sum_out = self.half_adder_2_xor(torch.cat([sum1, cin], dim=1))
        carry2 = self.half_adder_2_and(torch.cat([sum1, cin], dim=1))

        # Final Carry
        cout = self.or_gate(torch.cat([carry1, carry2], dim=1))
        return torch.cat([sum_out, cout], dim=1)

class MUX2(nn.Module):
    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        if implementation == "threshold":
            gate = create_threshold_gate
        elif implementation == "sigmoid":
            gate = lambda name: create_sigmoid_gate(name, sharpness=sharpness)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

        self.not_gate = gate("NOT")
        self.and_gate_1 = gate("AND")
        self.and_gate_2 = gate("AND")
        self.or_gate = gate("OR")

    def forward(self, x):
        d0 = x[:, 0:1]
        d1 = x[:, 1:2]
        s = x[:, 2:3]
        not_s = self.not_gate(s)
        left = self.and_gate_1(torch.cat([d0, not_s], dim=1))
        right = self.and_gate_2(torch.cat([d1, s], dim=1))
        return self.or_gate(torch.cat([left, right], dim=1))

class MUX4(nn.Module):

    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        self.mux_low = MUX2(implementation=implementation, sharpness=sharpness)
        self.mux_high = MUX2(implementation=implementation, sharpness=sharpness)
        self.mux_final = MUX2(implementation=implementation, sharpness=sharpness)

    def forward(self, x):
        # Inputs
        d0 = x[:, 0:1]
        d1 = x[:, 1:2]
        d2 = x[:, 2:3]
        d3 = x[:, 3:4]
        s0 = x[:, 4:5]
        s1 = x[:, 5:6]

        # MUX Low
        low = self.mux_low(torch.cat([d0, d1, s0], dim=1))
        high = self.mux_high(torch.cat([d2, d3, s0], dim=1))
        out = self.mux_final(torch.cat([low, high, s1], dim=1))
        return out