import torch
import torch.nn as nn

from gates import create_composed_gate, create_sigmoid_gate, create_threshold_gate


def _gate_factory(implementation, sharpness):
    """Return a single-argument gate constructor for the given implementation."""
    if implementation == "threshold":
        return create_threshold_gate
    if implementation == "sigmoid":
        return lambda name: create_sigmoid_gate(name, sharpness=sharpness)
    raise ValueError(f"Unknown implementation: {implementation}")

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


class PriorityEncoder(nn.Module):
    """4-to-2 priority encoder built from primitive gates.

    Inputs i0..i3 with i3 the highest priority. Emits the 2-bit index of the
    highest active input together with a valid flag: [a1, a0, valid].

        a1    = i2 OR i3
        a0    = i3 OR (i1 AND NOT i2)
        valid = i0 OR i1 OR i2 OR i3
    """

    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        gate = _gate_factory(implementation, sharpness)
        self.not_i2 = gate("NOT")
        self.and_a0 = gate("AND")
        self.or_a0 = gate("OR")
        self.or_a1 = gate("OR")
        self.or_valid_low = gate("OR")
        self.or_valid_high = gate("OR")
        self.or_valid = gate("OR")

    def forward(self, x):
        i0 = x[:, 0:1]
        i1 = x[:, 1:2]
        i2 = x[:, 2:3]
        i3 = x[:, 3:4]

        a1 = self.or_a1(torch.cat([i2, i3], dim=1))
        not_i2 = self.not_i2(i2)
        a0_term = self.and_a0(torch.cat([i1, not_i2], dim=1))
        a0 = self.or_a0(torch.cat([i3, a0_term], dim=1))

        valid_low = self.or_valid_low(torch.cat([i0, i1], dim=1))
        valid_high = self.or_valid_high(torch.cat([i2, i3], dim=1))
        valid = self.or_valid(torch.cat([valid_low, valid_high], dim=1))

        return torch.cat([a1, a0, valid], dim=1)


class Comparator1Bit(nn.Module):
    """1-bit magnitude comparator. Outputs [a>b, a==b, a<b]."""

    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        gate = _gate_factory(implementation, sharpness)
        self.not_a = gate("NOT")
        self.not_b = gate("NOT")
        self.and_gt = gate("AND")
        self.xnor_eq = gate("XNOR")
        self.and_lt = gate("AND")

    def forward(self, x):
        a = x[:, 0:1]
        b = x[:, 1:2]
        not_a = self.not_a(a)
        not_b = self.not_b(b)
        gt = self.and_gt(torch.cat([a, not_b], dim=1))
        eq = self.xnor_eq(torch.cat([a, b], dim=1))
        lt = self.and_lt(torch.cat([not_a, b], dim=1))
        return torch.cat([gt, eq, lt], dim=1)


class Comparator2Bit(nn.Module):
    """2-bit magnitude comparator.

    A = (a1, a0) and B = (b1, b0) are unsigned 2-bit numbers. Outputs the
    three relations [A>B, A==B, A<B] using bit-level equality/greater terms:

        eq = (a1 == b1) AND (a0 == b0)
        gt = (a1 AND NOT b1) OR ((a1 == b1) AND (a0 AND NOT b0))
        lt = (NOT a1 AND b1) OR ((a1 == b1) AND (NOT a0 AND b0))
    """

    def __init__(self, implementation="threshold", sharpness=20.0):
        super().__init__()
        gate = _gate_factory(implementation, sharpness)
        self.not_a1 = gate("NOT")
        self.not_b1 = gate("NOT")
        self.not_a0 = gate("NOT")
        self.not_b0 = gate("NOT")
        self.xnor_hi = gate("XNOR")
        self.xnor_lo = gate("XNOR")
        self.and_eq = gate("AND")
        self.and_gt_hi = gate("AND")
        self.and_gt_lo = gate("AND")
        self.and_gt_lo_eq = gate("AND")
        self.or_gt = gate("OR")
        self.and_lt_hi = gate("AND")
        self.and_lt_lo = gate("AND")
        self.and_lt_lo_eq = gate("AND")
        self.or_lt = gate("OR")

    def forward(self, x):
        a1 = x[:, 0:1]
        a0 = x[:, 1:2]
        b1 = x[:, 2:3]
        b0 = x[:, 3:4]

        not_a1 = self.not_a1(a1)
        not_b1 = self.not_b1(b1)
        not_a0 = self.not_a0(a0)
        not_b0 = self.not_b0(b0)

        eq_hi = self.xnor_hi(torch.cat([a1, b1], dim=1))
        eq_lo = self.xnor_lo(torch.cat([a0, b0], dim=1))
        eq = self.and_eq(torch.cat([eq_hi, eq_lo], dim=1))

        gt_hi = self.and_gt_hi(torch.cat([a1, not_b1], dim=1))
        gt_lo = self.and_gt_lo(torch.cat([a0, not_b0], dim=1))
        gt_lo_eq = self.and_gt_lo_eq(torch.cat([eq_hi, gt_lo], dim=1))
        gt = self.or_gt(torch.cat([gt_hi, gt_lo_eq], dim=1))

        lt_hi = self.and_lt_hi(torch.cat([not_a1, b1], dim=1))
        lt_lo = self.and_lt_lo(torch.cat([not_a0, b0], dim=1))
        lt_lo_eq = self.and_lt_lo_eq(torch.cat([eq_hi, lt_lo], dim=1))
        lt = self.or_lt(torch.cat([lt_hi, lt_lo_eq], dim=1))

        return torch.cat([gt, eq, lt], dim=1)