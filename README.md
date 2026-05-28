# Neural Network Blocks

This repository generates a database of small neural networks that implement basic Boolean logic gates and circuit-level building blocks.

# Features

* Symbolic truth-table generation
* Exact threshold-based logic networks
* Differentiable sigmoid-based logic networks
* Trainable MLP logic networks
* Compositional neural circuits (e.g. XOR)
* Exhaustive truth-table verification
* Structured JSON database generation
* Modular architecture for future extensions

---

# Repository Structure

```text
neural-network-blocks/
│
├── generate_database.py
│
├── src/
│   └── neural_network_blocks/
│       ├── truth_tables.py
│       ├── models.py
│       ├── gates.py
│       ├── verify.py
│       └── database.py
│
├── database/
│   └── boolean/
│
├── tests/
│   └── test_gates.py
│
├── requirements.txt
└── README.md
```

---

# Architecture Overview

The project is divided into three conceptual layers:

| Layer              | Responsibility                  |
| ------------------ | ------------------------------- |
| Symbolic Layer     | Defines logical behavior        |
| Neural Layer       | Implements logic in neural form |
| Verification Layer | Validates symbolic equivalence  |

Pipeline:

```text
Symbolic Logic
      ↓
Truth Table
      ↓
Neural Implementations
      ↓
Verification
      ↓
Database Entry
```

---

# Supported Logic Gates

Current Boolean logic support:

* NOT
* AND
* OR
* NAND
* NOR
* XOR
* XNOR
* IMPLIES
* EQUIVALENCE

---

# Neural Implementations

## 1. Threshold Networks

Single-neuron perceptrons implementing exact Boolean logic.

Example:

```text
AND(x1,x2) = step(x1 + x2 - 1.5)
```

Properties:

* exact
* interpretable
* fully verifiable

---

## 2. Sigmoid Networks

Differentiable approximations of logical functions.

Example:

```text
y = sigmoid(k(Wx+b))
```

Properties:

* smooth
* differentiable
* gradient-trainable

---

## 3. MLP Networks

Trainable multilayer perceptrons that learn logic from truth tables.

Architecture:

```text
Input
 ↓
Linear
 ↓
Tanh
 ↓
Linear
 ↓
Sigmoid
```

Properties:

* learned representations
* trainable
* extensible

---

## 4. Compositional Neural Circuits

Hierarchical neural circuits built from reusable submodules.

Example:

```text
XOR =
AND(
    OR(x1,x2),
    NAND(x1,x2)
)
```

Properties:

* modular
* reusable
* scalable

---

# Installation

## Clone Repository

```bash
git clone "https://github.com/mohanasrujana/neural-network-blocks.git"
cd neural-network-blocks
```

---

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Requirements

Core dependencies:

```text
torch
numpy
```

Recommended Python version:

```text
Python 3.10+
```

---

# Usage

## Generate Database

Run:

```bash
python generate_database.py
```

This generates:

* symbolic truth tables
* neural implementations
* verification metadata
* JSON database entries

Outputs are stored in:

```text
database/boolean/
```

---

# Example Output

Generated files:

```text
database/
└── boolean/
    ├── AND.json
    ├── OR.json
    ├── XOR.json
    └── weights/
```

---

# Example Database Entry

```json
{
  "name": "AND",
  "logic_type": "boolean",
  "truth_table": [...],
  "implementations": [...]
}
```

Each entry stores:

* symbolic specification
* architecture metadata
* neural parameters
* verification results

---

# Module Documentation

---

# `truth_tables.py`

Defines symbolic logic specifications.

## Responsibilities

* Generate binary input combinations
* Define symbolic gate behavior
* Construct truth tables

## Main Functions

| Function                   | Description                       |
| -------------------------- | --------------------------------- |
| `generate_binary_inputs()` | Generates all binary combinations |
| `gate_function()`          | Defines symbolic gate behavior    |
| `get_gate_arity()`         | Returns number of inputs          |
| `generate_truth_table()`   | Builds complete truth table       |

---

# `models.py`

Defines neural-network implementations of logic.

## Classes

| Class           | Description                     |
| --------------- | ------------------------------- |
| `ThresholdGate` | Exact threshold perceptron      |
| `SigmoidGate`   | Differentiable sigmoid neuron   |
| `MLPGate`       | Trainable multilayer perceptron |
| `ComposedXOR`   | Hierarchical XOR composition    |
| `ComposedXNOR`  | Hierarchical XNOR composition   |

---

# `gates.py`

Factory module for neural gate creation.

## Responsibilities

* Store exact gate parameters
* Create threshold implementations
* Create sigmoid implementations
* Create compositional circuits

## Main Functions

| Function                  | Description                    |
| ------------------------- | ------------------------------ |
| `create_threshold_gate()` | Creates threshold gate         |
| `create_sigmoid_gate()`   | Creates sigmoid gate           |
| `create_composed_gate()`  | Creates compositional circuits |
| `get_exact_params()`      | Returns stored parameters      |

---

# `verify.py`

Handles:

* training
* verification
* tensor conversion
* serialization support

## Main Functions

| Function                             | Description                      |
| ------------------------------------ | -------------------------------- |
| `truth_table_to_tensors()`           | Converts truth tables to tensors |
| `verify_model_against_truth_table()` | Exhaustive verification          |
| `train_mlp()`                        | Trains MLP from truth table      |
| `extract_state_dict_as_lists()`      | Serializes weights               |

---

# `database.py`

Creates structured JSON database entries.

## Responsibilities

* Store symbolic specifications
* Store neural metadata
* Store verification results
* Save reusable neural blocks

---

# `generate_database.py`

Coordinates the full database-generation pipeline.

## Pipeline

```text
Gate
 ↓
Truth Table
 ↓
Threshold Model
 ↓
Sigmoid Model
 ↓
MLP Training
 ↓
Verification
 ↓
JSON Export
```

---

# Verification

Verification is exhaustive for Boolean logic.

For a gate with `n` inputs:

```text
2^n
```

possible input combinations are evaluated.

Verification ensures:

* symbolic equivalence
* exact correctness
* deterministic behavior

---

# Design Principles

## Modular Architecture

Each module has a single responsibility:

* symbolic logic
* neural implementation
* verification
* storage

---

## Separation of Concerns

The project intentionally separates:

* specification
* implementation
* verification

This improves:

* maintainability
* extensibility
* interpretability

---

## Reusability

Neural blocks are treated as reusable computational primitives rather than isolated models.

---

## Compositionality

Complex logic is constructed hierarchically from smaller verified components.

---

# Development Workflow

## Run Database Generation

```bash
python generate_database.py
```

---

## Run Tests

```bash
pytest
```

---

## Add New Gates

1. Add symbolic definition in `truth_tables.py`
2. Add exact parameters in `gates.py`
3. Add compositional logic if needed
4. Regenerate database

---

# Current Limitations

* Boolean logic only
* Small-scale circuits
* Exhaustive verification scales exponentially
* No temporal logic support yet
* No first-order logic grounding yet

---

# Planned Extensions

## Near-Term

* Half adders
* Full adders
* Multiplexers
* Comparators
* Majority circuits

---

## Long-Term

* First-order logic grounding
* Temporal logic
* Neural-to-logic inversion
* Composition graphs
* Residual learning modules
* Circuit-level verification

---

# Future Vision

Long-term goal:

```text
Symbolic Logic
      ↓
Verified Neural Blocks
      ↓
Composable Neural Circuits
      ↓
Reusable Certified Neural Systems
```

The database is intended to evolve into reusable neural-symbolic infrastructure for scalable verified AI systems.


