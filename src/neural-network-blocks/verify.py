import torch
from .truth_tables import generate_truth_table


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


def train_mlp(model, gate_name, epochs=2000, lr=0.05):
    truth_table = generate_truth_table(gate_name)
    X, y = truth_table_to_tensors(truth_table)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

    verification = verify_model_against_truth_table(model, gate_name)

    return model, verification


def extract_state_dict_as_lists(model):
    state = model.state_dict()
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in state.items()
    }
