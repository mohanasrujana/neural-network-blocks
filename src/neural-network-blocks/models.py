import torch
import torch.nn as nn

class ThresholdGate(nn.Module):
    """
    Exact threshold neural gate.
    Output is 0 or 1.
    """

    def __init__(self, weights, bias):
        super().__init__()
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32).view(-1, 1))
        self.register_buffer("bias", torch.as_tensor([bias], dtype=torch.float32))

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        z = x @ self.weights + self.bias
        return (z >= 0).float()

class SigmoidGate(nn.Module):
    """
    Differentiable approximation of a threshold gate.
    """

    def __init__(self, weights, bias, sharpness=20.0):
        super().__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.bias = torch.tensor([bias], dtype=torch.float32)
        self.sharpness = sharpness

    def forward(self, x):
        z = x @ self.weights.view(-1, 1) + self.bias
        return torch.sigmoid(self.sharpness * z)



class MLPGate(nn.Module):
    """
    Trainable MLP for learning a logic gate from its truth table.
    """

    def __init__(self, input_dim, hidden_dim=4, output_dim=1):
        """
        Args:
            input_dim: number of input features
            hidden_dim: number of hidden units
            output_dim: number of output units
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class ComposedXOR(nn.Module):
    """
    XOR = AND(OR(x1, x2), NAND(x1, x2))
    This is a compositional neural logic block.
    """

    def __init__(self, or_gate, nand_gate, and_gate):
        super().__init__()
        self.or_gate = or_gate
        self.nand_gate = nand_gate
        self.and_gate = and_gate

    def forward(self, x):
        h1 = self.or_gate(x)
        h2 = self.nand_gate(x)
        h = torch.cat([h1, h2], dim=1)
        return self.and_gate(h)


class ComposedXNOR(nn.Module):
    """
    XNOR = NOT(XOR)
    """

    def __init__(self, xor_gate, not_gate):
        super().__init__()
        self.xor_gate = xor_gate
        self.not_gate = not_gate

    def forward(self, x):
        y = self.xor_gate(x)
        return self.not_gate(y)

