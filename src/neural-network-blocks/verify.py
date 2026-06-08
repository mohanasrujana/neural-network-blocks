import random

import torch
import torch.nn as nn

from truth_tables import generate_truth_table, extract_variables, program_truth_table
from models import MLPGate

DEFAULT_MLP_BASE_SEED = 42
DEFAULT_MLP_MAX_RETRIES = 5


def gate_training_seed(gate_name, base_seed=DEFAULT_MLP_BASE_SEED):
    """Stable per-gate seed (independent of Python's salted hash)."""
    return base_seed + sum(ord(c) for c in gate_name.upper())


def set_training_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def reset_module_parameters(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def truth_table_to_tensors(truth_table):
    X = torch.tensor([row["input"] for row in truth_table], dtype=torch.float32)
    y = torch.tensor([row["output"] for row in truth_table], dtype=torch.float32)
    return X, y


def binarize_output(y_pred, threshold=0.5):
    return (y_pred >= threshold).float()


def verify_model_against_truth_table(model, gate_name, threshold=0.5):
    truth_table = generate_truth_table(gate_name)
    X, y_true = truth_table_to_tensors(truth_table)

    with torch.no_grad():
        y_pred_raw = model(X)
        y_pred = binarize_output(y_pred_raw, threshold=threshold)

    correct = (y_pred == y_true).all(dim=1)
    accuracy = correct.float().mean().item()
    verified = bool(accuracy == 1.0)

    results = []
    for i, row in enumerate(truth_table):
        results.append({
            "input": row["input"],
            "expected": row["output"],
            "predicted_raw": y_pred_raw[i].detach().numpy().tolist(),
            "predicted_binary": y_pred[i].detach().numpy().astype(int).tolist(),
            "correct": bool(correct[i].item())
        })

    return {
        "verified": verified,
        "accuracy": accuracy,
        "num_cases": len(truth_table),
        "results": results
    }


def _train_mlp_once(model, X, y, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()


def train_mlp(
    model,
    gate_name,
    epochs=2000,
    lr=0.05,
    base_seed=DEFAULT_MLP_BASE_SEED,
    max_retries=DEFAULT_MLP_MAX_RETRIES,
):
    truth_table = generate_truth_table(gate_name)
    X, y = truth_table_to_tensors(truth_table)
    gate_seed = gate_training_seed(gate_name, base_seed)

    last_verification = None
    for attempt in range(max_retries):
        set_training_seed(gate_seed + attempt)
        reset_module_parameters(model)
        _train_mlp_once(model, X, y, epochs, lr)
        verification = verify_model_against_truth_table(model, gate_name)
        if verification["verified"]:
            return model, verification
        last_verification = verification

    raise RuntimeError(
        f"MLP training failed for {gate_name} after {max_retries} attempts "
        f"(seeds {gate_seed}..{gate_seed + max_retries - 1}, "
        f"last accuracy: {last_verification['accuracy']:.4f})"
    )


def extract_state_dict_as_lists(model):
    state = model.state_dict()
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in state.items()
    }

def save_model_state_dict(model, path):
    """Save a model's parameters in native PyTorch format (``torch.save``).
    Returns the path as a string so callers can record it in the database.
    """
    torch.save(model.state_dict(), path)
    return str(path)

def verify_program(model, expression):
    table = program_truth_table(expression)
    variables = extract_variables(expression)
    X,y = [],[]

    for row in table:
        X.append([row["input"][v] for v in variables])
        y.append([row["output"]])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    pred = model(X)
    pred = (pred > 0.5).float()
    matches = (pred == y)

    return {
        "verified": bool(matches.all()),
        "accuracy": matches.float().mean().item(),
        "num_cases":len(table)
    }


def program_to_tensors(expression):
    table = program_truth_table(expression)
    variables = extract_variables(expression)
    X = torch.tensor(
        [[float(row["input"][v]) for v in variables] for row in table],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [[float(row["output"])] for row in table],
        dtype=torch.float32,
    )
    return X, y, variables, table


def verify_program_mlp(model, X, y, num_cases, threshold=0.5):
    with torch.no_grad():
        pred = (model(X) > threshold).float()
    matches = (pred == y)
    return {
        "verified": bool(matches.all()),
        "accuracy": matches.float().mean().item(),
        "num_cases": num_cases,
    }


def _fold_input_scale(model, scale):
    """Bake a per-feature input scaling into the first linear layer.

    The MLP trains on inputs normalised to roughly [0, 1] for stable
    optimisation, but the stored model should accept the raw integer inputs
    used everywhere else. Since W @ (x / s) == (W / s) @ x, dividing the first
    layer's weight columns by the scale yields an equivalent raw-input model.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.div_(scale.view(1, -1))
                break


def train_program_mlp(
    name,
    expression,
    hidden_dim=8,
    epochs=4000,
    lr=0.05,
    base_seed=DEFAULT_MLP_BASE_SEED,
    max_retries=DEFAULT_MLP_MAX_RETRIES,
):
    X, y, variables, table = program_to_tensors(expression)
    scale = X.abs().max(dim=0).values.clamp(min=1.0)
    X_norm = X / scale
    seed = gate_training_seed(name, base_seed)

    last_verification = None
    for attempt in range(max_retries):
        set_training_seed(seed + attempt)
        model = MLPGate(input_dim=len(variables), hidden_dim=hidden_dim, output_dim=1)
        reset_module_parameters(model)
        _train_mlp_once(model, X_norm, y, epochs, lr)
        _fold_input_scale(model, scale)
        verification = verify_program_mlp(model, X, y, len(table))
        if verification["verified"]:
            return model, verification, variables
        last_verification = verification

    raise RuntimeError(
        f"Program MLP training failed for '{name}' ({expression}) after "
        f"{max_retries} attempts (last accuracy: {last_verification['accuracy']:.4f})"
    )