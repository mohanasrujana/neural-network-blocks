# Neural Network Blocks

This repository generates a database of small neural networks that implement basic Boolean logic gates and compositional circuit blocks. It also includes a symbolic layer for parsing and evaluating Boolean programs and comparison expressions.

## Features

- Symbolic truth-table generation for gates and circuits
- Exact threshold-based logic networks
- Differentiable sigmoid-based logic networks
- Trainable MLP logic networks
- Compositional neural circuits (XOR, XNOR, half/full adders, multiplexers, priority encoder, comparators)
- Boolean expression parsing and evaluation
- Exhaustive truth-table verification
- Structured JSON database generation

---

## Repository Structure

```text
neural-network-blocks/
│
├── generate_database.py          # Database generation pipeline
│
├── src/
│   └── neural-network-blocks/
│       ├── truth_tables.py       # Gate truth tables and expression evaluation
│       ├── models.py             # Threshold, sigmoid, MLP, and composed gate models
│       ├── gates.py              # Gate factory and exact parameters
│       ├── circuits.py           # Compositional circuit modules
│       ├── verify.py             # Training, verification, serialization
│       ├── database.py           # JSON database entry builders
│       ├── parser.py             # Boolean/comparison expression parser
│       ├── logic_graph.py        # AST node types for parsed expressions
│       ├── evaluator.py          # Symbolic evaluation of logic graphs
│       └── program_examples.py   # Example Boolean programs
│
├── database/
│   └── boolean/                  # Generated gate/circuit entries and MLP weights
│
├── requirements.txt
└── README.md
```

---

## Architecture Overview

The project is divided into three conceptual layers:

| Layer              | Responsibility                                      |
| ------------------ | --------------------------------------------------- |
| Symbolic Layer     | Defines logical behavior and parses expressions     |
| Neural Layer       | Implements logic in neural form                     |
| Verification Layer | Validates symbolic equivalence via truth tables     |

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

## Supported Gates and Circuits

### Boolean Gates

- NOT
- AND
- OR
- NAND
- NOR
- XOR
- XNOR
- IMPLIES
- EQUIVALENCE

### Compositional Circuits

| Circuit          | Inputs | Outputs           | Description                                      |
| ---------------- | ------ | ----------------- | ------------------------------------------------ |
| HALF_ADDER       | 2      | sum, carry        | XOR + AND composition                            |
| FULL_ADDER       | 3      | sum, carry        | Two half adders + OR carry chain                 |
| MUX2             | 3      | 1                 | 2-to-1 multiplexer (d0, d1, select)              |
| MUX4             | 6      | 1                 | 4-to-1 multiplexer (d0–d3, s0, s1)               |
| PRIORITY_ENCODER | 4      | a1, a0, valid     | 4-to-2 priority encoder (i3 highest priority)    |
| COMPARATOR_1BIT  | 2      | gt, eq, lt        | 1-bit magnitude comparator (a>b, a==b, a<b)      |
| COMPARATOR_2BIT  | 4      | gt, eq, lt        | 2-bit magnitude comparator (A vs B)              |

---

## Neural Implementations

Each gate or circuit in the database can include up to three implementation types.

### 1. Threshold Networks

Single-neuron perceptrons implementing exact Boolean logic.

```text
AND(x1, x2) = step(x1 + x2 - 1.5)
```

- Exact and interpretable
- Fully verifiable against truth tables
- Used for primitive gates and compositional circuits (XOR, half adder)

### 2. Sigmoid Networks

Differentiable approximations of logical functions.

```text
y = sigmoid(k(Wx + b))
```

- Smooth and gradient-friendly
- Verified after thresholding at 0.5
- Available for gates and the half adder

### 3. MLP Networks

Trainable multilayer perceptrons that learn logic from truth tables.

```text
Input → Linear → Tanh → Linear → Sigmoid
```

- Trained for 3000 epochs with learning rate 0.05
- Weights saved separately under `database/boolean/weights/`
- Generated for all gates and circuits, including full adder and multiplexers

### 4. Compositional Neural Circuits

Higher-level modules built from verified sub-gates in `circuits.py`:

```text
XOR = AND(OR(x1, x2), NAND(x1, x2))
HALF_ADDER = [XOR(a, b), AND(a, b)]
FULL_ADDER = two half adders + OR carry
MUX2 / MUX4 = AND/OR/NOT multiplexer trees
PRIORITY_ENCODER = OR/AND/NOT priority resolution tree
COMPARATOR_1BIT / COMPARATOR_2BIT = AND/OR/NOT/XNOR relation network
```

Every compositional circuit is emitted with all three implementation types: exact threshold, differentiable sigmoid, and a trained MLP.

---

## Symbolic Expression Layer

Beyond fixed gate definitions, the codebase supports parsing and evaluating Boolean programs and comparison expressions.

**Parser** (`parser.py`) — converts expressions like `(a AND b) OR c` or `x > 5` into a logic graph.

**Evaluator** (`evaluator.py`) — evaluates the graph over variable assignments.

**Truth tables** (`truth_tables.py`) — utilities for:

- `generate_program_truth_table(expression)` — exhaustive Boolean program tables
- `generate_equation_truth_table(expression, variable_domains)` — tables over numeric domains
- `extract_variables(expression)` — variable discovery

Example programs are defined in `program_examples.py`:

```python
PROGRAMS = {
    "launch": "(a AND b) OR c",
    "alarm": "(sensor AND door) AND NOT override",
    "greater_than_5": "x > 5",
    "between_4_and_9": "(x > 4) AND (x < 9)",
}
```

---

## Installation

### Clone Repository

```bash
git clone "https://github.com/mohanasrujana/neural-network-blocks.git"
cd neural-network-blocks
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Requirements:** `torch`, `numpy`  
**Recommended:** Python 3.10+

---

## Usage

### Generate Database

```bash
python generate_database.py
```

This generates, for each gate and circuit:

- Symbolic truth tables
- Threshold and/or sigmoid implementations (where applicable)
- Trained MLP implementations with saved weights
- Verification metadata

Output locations:

```text
database/boolean/
├── AND.json
├── OR.json
├── XOR.json
├── HALF_ADDER.json
├── FULL_ADDER.json
├── MUX2.json
├── MUX4.json
└── weights/
    ├── and_mlp_weights.json
    ├── half_adder_mlp_weights.json
    └── ...
```

### Example Database Entry

```json
{
  "name": "AND",
  "logic_type": "boolean",
  "description": "AND Boolean logic gate",
  "inputs": ["x1", "x2"],
  "outputs": ["y"],
  "truth_table": [...],
  "implementations": [
    { "implementation_type": "exact_threshold", ... },
    { "implementation_type": "differentiable_sigmoid", ... },
    { "implementation_type": "trained_mlp", ... }
  ]
}
```

Each entry stores the symbolic specification, architecture metadata, neural parameters (or weight file paths), and verification results.

---

## Module Reference

| Module               | Role                                                                 |
| -------------------- | -------------------------------------------------------------------- |
| `truth_tables.py`    | Gate truth tables, arity, expression-based table generation          |
| `models.py`          | `ThresholdGate`, `SigmoidGate`, `MLPGate`, `ComposedXOR`, `ComposedXNOR` |
| `gates.py`           | Factory functions and exact perceptron parameters                    |
| `circuits.py`        | `HalfAdder`, `FullAdder`, `MUX2`, `MUX4`, `PriorityEncoder`, `Comparator1Bit`, `Comparator2Bit` compositional modules |
| `verify.py`          | Tensor conversion, exhaustive verification, MLP training             |
| `database.py`        | JSON entry builders and file export                                  |
| `parser.py`          | AST-based expression parsing                                         |
| `logic_graph.py`     | Node types for variables, constants, gates, comparisons              |
| `evaluator.py`       | Recursive graph evaluation                                           |
| `program_examples.py`| Sample Boolean and comparison programs                               |

---

## Verification

Verification is exhaustive for Boolean gates and circuits. For a block with `n` binary inputs, all `2^n` input combinations are evaluated.

Each implementation records:

- `verified` — whether all cases match the symbolic truth table
- `accuracy` — fraction of matching cases
- `num_cases` — total cases tested

---

## Design Principles

- **Modular architecture** — symbolic logic, neural implementation, verification, and storage are separate concerns
- **Reusability** — neural blocks are treated as composable computational primitives
- **Compositionality** — complex logic is built hierarchically from smaller verified components

---

## Development Workflow

### Add a New Gate or Circuit

1. Add symbolic behavior and arity in `truth_tables.py`
2. Add exact parameters in `gates.py` (if a single-neuron gate)
3. Add a compositional module in `circuits.py` (if a multi-gate circuit)
4. Register the name in `GATES` inside `generate_database.py`
5. Regenerate the database

### Regenerate Database

```bash
python generate_database.py
```

---

## Current Limitations

- Boolean logic and small-scale circuits only
- Exhaustive verification scales exponentially with input count
- Expression layer supports parsing and symbolic evaluation but is not yet wired into neural training
- No automated test suite in the repository

---

## Planned Extensions

- Majority circuits and wider encoders/decoders
- Neural training from parsed Boolean programs
- First-order logic grounding
- Temporal logic support
- Neural-to-logic inversion and composition graphs
- Circuit-level verification beyond truth-table equivalence

---

## Future Vision

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
