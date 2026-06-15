# Code Workflows

This document explains how the main code paths in this repository work. The project has two primary generation workflows:

- Boolean gate/circuit workflow.
- Symbolic program workflow.

Both workflows follow the same research idea:

```text
Symbolic specification
  -> truth table
  -> neural implementation
  -> verification
  -> JSON database
  -> visualization
```

## Repository Map

Core source files:

- `truth_tables.py` defines symbolic truth tables and program truth-table generation.
- `models.py` defines neural modules: threshold gates, sigmoid gates, and MLPs.
- `gates.py` maps primitive gate names to exact threshold/sigmoid parameters.
- `circuits.py` builds larger circuits by composing smaller gates.
- `parser.py` parses expressions and lowers statement-style programs into logic graphs.
- `logic_graph.py` defines graph node types.
- `evaluator.py` evaluates logic graphs symbolically.
- `compiler.py` evaluates logic graphs as neural programs.
- `verify.py` verifies models against truth tables and trains MLPs.
- `database.py` builds JSON database entries.
- `program_examples.py` contains the symbolic programs.
- `visualize_network.py` renders JSON/database artifacts into SVG and HTML.

Top-level generators:

- `generate_database.py` generates Boolean gate/circuit database entries.
- `generate_program_database.py` generates symbolic program database entries.

Generated outputs:

- `database/boolean/`
- `database/programs/`
- `database/boolean/weights/`
- `database/programs/weights/`
- `visualizations/`

## Workflow 1: Boolean Gate and Circuit Database

Entry point:

```bash
python generate_database.py
```

This workflow handles primitive gates and composed circuits such as:

- `AND`
- `OR`
- `XOR`
- `XNOR`
- `HALF_ADDER`
- `FULL_ADDER`
- `MUX2`
- `MUX4`
- `PRIORITY_ENCODER`
- `COMPARATOR_1BIT`
- `COMPARATOR_2BIT`

### Step 1: Select Registered Block

`generate_database.py` defines the block list in `GATES`.

For each gate/circuit name, it calls:

```python
generate_gate_entry(gate_name)
```

### Step 2: Build Truth Table

Truth tables come from `truth_tables.py`.

Main functions:

- `get_gate_arity(name)`
- `generate_binary_inputs(n_inputs)`
- `gate_function(name, x)`
- `generate_truth_table(name)`

The truth table is exhaustive over binary inputs.

Example:

```text
AND has 2 inputs -> 4 cases
FULL_ADDER has 3 inputs -> 8 cases
MUX4 has 6 inputs -> 64 cases
```

### Step 3: Build Exact Threshold or Compositional Model

Primitive single-neuron gate parameters live in `gates.py`:

```python
EXACT_GATE_PARAMS = {
    "AND": {"weights": [1.0, 1.0], "bias": -1.5},
    ...
}
```

Primitive gates use:

```python
create_threshold_gate(name)
```

Some gates cannot be represented as a single threshold neuron, such as `XOR`, so they use composed gates:

```python
create_composed_gate("XOR")
```

Larger circuits use classes in `circuits.py`, for example:

```python
HalfAdder()
FullAdder()
MUX2()
Comparator2Bit()
```

### Step 4: Build Sigmoid Model

The sigmoid model follows the same logical structure as the threshold model, but replaces hard threshold activations with sigmoid activations.

Primitive gates use:

```python
create_sigmoid_gate(name, sharpness=20.0)
```

Composed circuits use:

```python
CircuitClass(implementation="sigmoid", sharpness=20.0)
```

Sigmoid outputs are verified after thresholding at `0.5`.

### Step 5: Train MLP Model

Each gate/circuit also gets a trained MLP.

Model:

```python
MLPGate(input_dim, hidden_dim, output_dim)
```

Training happens in `verify.py`:

```python
train_mlp(model, gate_name, epochs=3000, lr=0.05)
```

Training details:

- Uses Adam.
- Uses binary cross-entropy loss.
- Uses stable deterministic seeds based on the gate name.
- Retries training up to `DEFAULT_MLP_MAX_RETRIES`.
- Fails if no retry reaches exact truth-table equivalence.

### Step 6: Verify Implementations

Verification uses:

```python
verify_model_against_truth_table(model, gate_name)
```

It:

1. Generates the symbolic truth table.
2. Converts truth-table rows into tensors.
3. Runs the neural model.
4. Binarizes outputs at `0.5`.
5. Compares predictions against expected truth-table outputs.

Verification output:

```json
{
  "verified": true,
  "accuracy": 1.0,
  "num_cases": 8,
  "results": [...]
}
```

### Step 7: Save Weights

MLP weights are saved in two formats:

- JSON state dict: portable and useful for visualizers.
- PyTorch state dict: directly loadable by PyTorch.

Functions:

```python
save_mlp_weights(gate_name, model)
save_mlp_model(gate_name, model)
```

Outputs:

```text
database/boolean/weights/<name>_mlp_weights.json
database/boolean/weights/<name>_mlp.pt
```

### Step 8: Build JSON Database Entry

`database.py` builds the JSON object:

```python
make_database_entry(gate_name, implementation_entries)
```

Each database entry includes:

- name,
- logic type,
- description,
- inputs,
- outputs,
- truth table,
- implementations.

Each implementation includes:

- implementation ID,
- implementation type,
- architecture metadata,
- weights or weight paths,
- verification metadata.

### Step 9: Write Database File

The final JSON is saved with:

```python
save_database_entry(entry)
```

Output:

```text
database/boolean/<GATE_NAME>.json
```

## Workflow 2: Symbolic Program Database

Entry point:

```bash
python generate_program_database.py
```

This workflow handles entries in `program_examples.py`.

Examples:

- Boolean expressions: `(a AND b) OR c`
- Comparison expressions: `x > 5`
- Arithmetic comparisons: `(x + y) > 10`
- Derived gates: `XOR(a, b)`
- Statement programs with assignments, branches, loops, and returns.

### Step 1: Iterate Over `PROGRAMS`

`generate_program_database.py` imports:

```python
from program_examples import PROGRAMS
```

Then it loops through:

```python
for name, expression in PROGRAMS.items():
    entry = generate_program_entry(name, expression)
```

### Step 2: Parse or Lower Program

Parsing happens through:

```python
parse_program(source)
```

in `parser.py`.

`parse_program` supports two modes:

- single expression mode,
- statement-program mode.

Expression examples:

```python
"x > 5"
"(a AND b) OR c"
"XOR(a, b)"
```

Statement example:

```python
steps = 0
v = x
while v > 0:
    v = v - 2
    steps = steps + 1
return steps == 4
```

### Step 3: Convert Syntax to Logic Graph

The parser returns:

```python
graph, variables = parse_program(expression)
```

The graph is built from node classes in `logic_graph.py`:

- `VariableNode`
- `ConstantNode`
- `GateNode`
- `ComparisonNode`
- `ArithmeticNode`

Example:

```text
(a AND b) OR c

becomes:

GateNode("OR", [
    GateNode("AND", [VariableNode("a"), VariableNode("b")]),
    VariableNode("c")
])
```

### Step 4: Lower Control Flow

Statement programs are lowered by `ProgramLowerer` in `parser.py`.

Supported statements:

- assignment,
- `if` / `else`,
- `for i in range(...)`,
- `while`,
- final `return`.

#### Assignments

Assignments store graph expressions in an environment.

Example:

```python
v = x + 1
```

stores:

```text
v -> ArithmeticNode("+", VariableNode("x"), ConstantNode(1))
```

#### If / Else

Branches are both symbolically executed, then merged with:

```text
condition * then_value + (1 - condition) * else_value
```

This creates a graph-level select operation using arithmetic nodes.

#### For Loops

Only constant `range(...)` loops are supported.

Example:

```python
for i in range(4):
    ...
```

The loop is unrolled at parse/lowering time.

#### While Loops

While loops are unrolled up to:

```python
WHILE_MAX_ITERATIONS = 11
```

Each iteration is guarded by the current loop condition, so updates only take effect when the condition is true.

This bound matches the default numeric input domain `0..10`.

### Step 5: Collect Free Variables

After graph construction, `collect_variables(graph)` walks the graph and returns free input variables.

This avoids treating local variables like `v`, `steps`, `count`, or loop variable `i` as external inputs.

Example:

```python
v = x
while v > 0:
    v = v - 1
return v == 0
```

Free variables:

```text
["x"]
```

not:

```text
["v", "x"]
```

### Step 6: Generate Program Truth Table

Truth-table generation uses:

```python
program_truth_table(expression)
```

in `truth_tables.py`.

It:

1. Calls `parse_program`.
2. Decides the input domain.
3. Enumerates all input combinations.
4. Evaluates the graph for each assignment.

Domain rule:

- If the expression contains a comparison operator, use integer domain `0..10`.
- Otherwise, use Boolean domain `{0, 1}`.

This is a finite-domain verification strategy, not proof over all integers.

### Step 7: Symbolic Evaluation

Symbolic graph evaluation uses:

```python
evaluate(graph, assignment)
```

in `evaluator.py`.

The evaluator handles:

- variables,
- constants,
- arithmetic,
- comparisons,
- gates.

It uses memoization:

```python
evaluate(node, values, memo=None)
```

Memoization caches node results by identity during one graph evaluation. This is important because lowered control-flow graphs reuse subgraphs heavily.

### Step 8: Compile Program to Neural Model

Program compilation uses:

```python
compile_program(expression, implementation="threshold")
compile_program(expression, implementation="sigmoid")
```

in `compiler.py`.

The compiled model is:

```python
CompiledProgram(graph, variables, implementation, sharpness)
```

At runtime, `CompiledProgram.forward(X)`:

1. Maps tensor columns to variable names.
2. Recursively evaluates the graph.
3. Uses neural gates for `GateNode`s.
4. Uses tensor arithmetic for arithmetic nodes.
5. Uses hard or soft comparisons for comparison nodes.

### Step 9: Threshold Compiled Program

For `implementation="threshold"`:

- logic gates use exact hard-threshold gates,
- comparisons use exact tensor comparisons,
- output is binary.

This is the exact compiled neural-program path.

### Step 10: Sigmoid Compiled Program

For `implementation="sigmoid"`:

- logic gates use sigmoid gates,
- comparisons use softened sigmoid comparisons,
- outputs are thresholded at `0.5` for verification.

The sigmoid comparison boundaries are shifted by `0.5` so integer comparisons match after thresholding.

Example:

```python
x > y
```

uses:

```python
sigmoid(k * (x - y - 0.5))
```

### Step 11: Verify Compiled Programs

Compiled threshold and sigmoid programs are verified by:

```python
verify_program(model, expression)
```

in `verify.py`.

It:

1. Generates the symbolic program truth table.
2. Builds tensor inputs in variable order.
3. Runs the compiled model.
4. Thresholds model output at `0.5`.
5. Checks every case against the truth table.

Output:

```json
{
  "verified": true,
  "accuracy": 1.0,
  "num_cases": 121
}
```

### Step 12: Train Program MLP

Program MLP training uses:

```python
train_program_mlp(name, expression, hidden_dim=8)
```

in `verify.py`.

It:

1. Converts the program truth table to tensors.
2. Normalizes inputs for training.
3. Trains an `MLPGate`.
4. Folds input normalization into the first linear layer.
5. Verifies the folded model on raw inputs.
6. Retries with deterministic seeds if needed.

The fold step matters because the stored model should accept raw integer inputs, even though training is easier with normalized inputs.

### Step 13: Save Program Weights

Program MLP weights are saved by:

```python
save_program_mlp_weights(name, model)
save_program_mlp_model(name, model)
```

Outputs:

```text
database/programs/weights/<program>_mlp_weights.json
database/programs/weights/<program>_mlp.pt
```

### Step 14: Build Program Database Entry

Program entries are built with:

```python
create_program_entry(name, expression, variables, truth_table, implementations)
```

Each program entry includes:

- program name,
- logic type,
- source expression/program,
- inputs,
- outputs,
- truth table,
- compiled threshold implementation,
- compiled sigmoid implementation,
- trained MLP implementation.

### Step 15: Write Program JSON

The final entry is written to:

```text
database/programs/<program_name>.json
```

## Workflow 3: Logic Graph Evaluation

The logic graph is the central intermediate representation for programs.

```text
source program
  -> parser/lowerer
  -> logic graph
  -> symbolic evaluator
  -> neural compiler
```

The graph uses a small set of node types:

```text
VariableNode
ConstantNode
GateNode
ComparisonNode
ArithmeticNode
```

The same graph can be used in two ways:

- Symbolically, through `evaluator.evaluate`.
- Neurally, through `compiler.CompiledProgram`.

That shared graph is what keeps the symbolic and neural paths aligned.

## Workflow 4: Verification

Verification is always based on comparing model outputs to symbolic truth tables.

### Gate/Circuit Verification

Function:

```python
verify_model_against_truth_table(model, gate_name)
```

Data source:

```python
generate_truth_table(gate_name)
```

Use cases:

- primitive threshold gates,
- primitive sigmoid gates,
- composed threshold circuits,
- composed sigmoid circuits,
- gate/circuit MLPs.

### Program Verification

Function:

```python
verify_program(model, expression)
```

Data source:

```python
program_truth_table(expression)
```

Use cases:

- compiled threshold programs,
- compiled sigmoid programs.

### Program MLP Verification

Function:

```python
verify_program_mlp(model, X, y, num_cases)
```

Use case:

- trained MLPs for symbolic programs.

## Workflow 5: Database Entry Creation

`database.py` is the serialization layer. It does not train or verify models. It packages already-computed metadata into JSON-friendly dictionaries.

Main Boolean/circuit entry functions:

- `make_database_entry`
- `make_threshold_implementation_entry`
- `make_sigmoid_implementation_entry`
- `make_mlp_implementation_entry`
- `save_database_entry`

Main program entry functions:

- `create_program_entry`
- `make_program_implementation_entry`
- `make_program_sigmoid_entry`
- `make_program_mlp_entry`

The database format keeps three things together:

- symbolic specification,
- neural implementation metadata,
- verification result.

## Workflow 6: Visualization

Visualization is generated from database artifacts and saved under `visualizations/`.

Main file:

```text
src/neural-network-blocks/visualize_network.py
```

The visualizer reconstructs diagrams for:

- exact threshold networks,
- sigmoid networks,
- trained MLPs,
- compiled program networks,
- schematic networks when full weights are not directly represented.

The generated SVGs use:

- blue edges for positive weights,
- red edges for negative weights,
- edge thickness proportional to absolute weight,
- labels for weights and biases,
- metadata for interactive inspection.

The generated `visualizations/index.html` provides:

- search,
- implementation-type filters,
- collapsible sections,
- modal diagram viewer,
- zoom controls,
- weight/bias toggles,
- connection detail panel.

## End-to-End Boolean/Circuit Flow

```text
generate_database.py
  -> GATES list
  -> truth_tables.generate_truth_table
  -> gates.py / circuits.py create model
  -> verify.verify_model_against_truth_table
  -> verify.train_mlp
  -> database.py create implementation entries
  -> database/boolean/<name>.json
  -> database/boolean/weights/*
  -> visualize_network.py
  -> visualizations/<name>/*.svg
```

## End-to-End Program Flow

```text
generate_program_database.py
  -> program_examples.PROGRAMS
  -> parser.parse_program
  -> parser.ProgramLowerer if statement program
  -> logic_graph nodes
  -> truth_tables.program_truth_table
  -> evaluator.evaluate
  -> compiler.compile_program(threshold)
  -> verify.verify_program
  -> compiler.compile_program(sigmoid)
  -> verify.verify_program
  -> verify.train_program_mlp
  -> database.py create program entry
  -> database/programs/<name>.json
  -> database/programs/weights/*
  -> visualize_network.py
  -> visualizations/<name>/*.svg
```

## Important Design Choices

### Finite-Domain Verification

Program verification is exhaustive over a chosen finite domain, not over all mathematical integers.

Current rule:

- Boolean programs: `{0, 1}`.
- Numeric/comparison programs: `0..10`.

This is practical for the prototype, but future versions should make domains explicit per program.

### Bounded While Loops

While loops are unrolled up to 11 iterations.

Reason:

- numeric inputs currently range over `0..10`,
- countdown-style programs need at most 11 guarded updates.

This keeps lowering finite and verifiable.

### Memoization

Memoization is used in both symbolic and neural graph evaluation.

Reason:

- lowered control-flow graphs share subgraphs,
- without caching, repeated subgraphs may be recomputed many times,
- memoization keeps evaluation practical for loop-heavy examples.

### Multiple Neural Implementations

The repository intentionally stores multiple implementations for the same symbolic behavior:

- exact threshold,
- differentiable sigmoid,
- trained MLP.

This supports comparison between interpretable, differentiable, and learned neural forms.

## Practical Developer Workflows

### Add a New Primitive Gate

1. Add logic to `truth_tables.gate_function`.
2. Add arity to `truth_tables.get_gate_arity`.
3. Add exact params to `gates.EXACT_GATE_PARAMS`, if single-threshold.
4. Add composed implementation if needed.
5. Register the name in `generate_database.py`.
6. Run `python generate_database.py`.
7. Regenerate visualizations.

### Add a New Circuit

1. Add symbolic behavior to `truth_tables.gate_function`.
2. Add arity to `truth_tables.get_gate_arity`.
3. Add a module in `circuits.py`.
4. Register it in `COMPOSITIONAL_CIRCUITS`.
5. Set output dimension if multi-output.
6. Run `python generate_database.py`.
7. Regenerate visualizations.

### Add a New Program

1. Add the source to `PROGRAMS` in `program_examples.py`.
2. Make sure it uses supported syntax.
3. Run `python generate_program_database.py`.
4. Check that all three implementations verify.
5. Regenerate visualizations.

Supported program syntax:

- Boolean keywords: `AND`, `OR`, `NOT`.
- Gate calls: `XOR(a, b)`, `NAND(a, b)`, etc.
- Comparisons: `>`, `<`, `>=`, `<=`, `==`, `!=`.
- Arithmetic: `+`, `-`, `*`, unary `-`.
- Assignment.
- `if` / `else`.
- `for ... in range(...)` with constant bounds.
- bounded `while`.
- final `return`.

### Debug a Program That Fails

Check in this order:

1. Does `parse_program` accept the syntax?
2. Are the free variables correct?
3. Does `program_truth_table` produce expected outputs?
4. Does `compile_program(..., "threshold")` verify?
5. Does `compile_program(..., "sigmoid")` verify?
6. Does `train_program_mlp` converge within retries?

Most bugs should be isolated before MLP training. If the symbolic truth table is wrong, training and verification will only reproduce the wrong target.

## Current Gaps

The workflows are functional, but the next engineering improvements should be:

- add tests,
- make program domains explicit,
- add a program diversity audit,
- package the source directory as a normal Python package,
- update README limitations,
- separate generated artifacts from source changes more cleanly.

