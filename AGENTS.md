# AGENTS.md - Build Agent Documentation for evolvenet

## Project Overview
**evolvenet** is a Go-based neural network simulator that uses evolutionary algorithms to train a network. The original XOR gate problem is solved through mutation, selection, and evolution over generations.

**Language**: Go 1.18
**Purpose**: Demonstrate neuroevolution for simple classification tasks

## Project Structure
The project was refactored from a single `main.go` file into modular components:

```
evolvenet/
├── go.mod                     # Go module definition
├── main.go                    # Entry point with init() and main()
├── utils.go                   # Utility functions (randFloat, roundFloat)
├── synapse.go                 # Synapse struct and methods (weights/connections)
├── neuron.go                  # Neuron struct and methods (activation, bias, synapses)
├── layer.go                   # Layer struct and methods (neuron collections)
├── network.go                 # Network struct and methods (forward pass, evaluation)
├── organism.go                # Organism struct and methods (population evolution)
```

## Commands

### Build
- **Command**: `go build`
- **Purpose**: Compile the project into a binary
- **Output**: `evolvenet` executable
- **Notes**: Run this after any code changes. No errors should occur.

### Run
- **Command**: `go run`
- **Purpose**: Execute the trained neural network simulation
- **Output**: Evolutionary training progress and final accuracy on XOR gate data
- **Expected**: Achieves 100% accuracy (TN: 2, TP: 2, Accuracy: 1.0)
- **Parameters**: Hardcoded for XOR training (2 inputs, 1 output, sigmoid activation)

### Test (if implemented)
- **Command**: `go test`
- **Purpose**: Run unit tests
- **Status**: No tests implemented yet - would require creating `*_test.go` files
- **Recommendation**: Add tests for each component (neuron activation, network forward pass, etc.)

### Lint & Format
- **Command**: `gofmt -l .` (check formatting)
- **Command**: `gofmt -w .` (apply formatting)
- **Purpose**: Ensure code follows Go formatting standards
- **Notes**: Already applied; code should pass without issues

### Dependencies
- **External**: None (uses only standard library)
- **Imports**: `fmt`, `math`, `math/rand`, `sort`, `strings`, `time`

## Development Workflow
When making changes:
1. **Lint/Format**: `gofmt -w .`
2. **Build**: `go build`
3. **Run**: `go run` (verify output matches expected accuracy)
4. **Iterate**: If issues, fix and repeat

## Key Components
- **Synapse**: Connection between neurons with weight and index
- **Neuron**: Processing unit with activation function, bias, and synapses
- **Layer**: Collection of neurons (input/hidden/output)
- **Network**: Full neural network with layers and loss calculation
- **Organism**: Population manager for evolutionary training
- **Utils**: Random generation and floating-point rounding

## Training Parameters
- **Data**: XOR gate truth table (4 samples)
- **Population**: 16 networks per generation
- **Generations**: Up to 10,000,000 (threshold 0.000001)
- **Mutation**: 10% rate on non-top performers
- **Selection**: Elitism + mutation
- **Logging**: Every 1000 generations

## Future Enhancements
- Add unit tests
- Parameterize training data/network architecture
- Add more activation functions
- Implement other loss functions
- Add serialization (save/load trained networks)
- Visualize evolution progress
