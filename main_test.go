package main

import (
	"fmt"
	"testing"
)

// TestANDGate tests evolution on AND gate logic (simpler than XOR)
func TestANDGate(t *testing.T) {
	network := &Network{}
	network.add_layer("input", "sigmoid", 2)
	network.add_layer("hidden", "sigmoid", 1)
	network.add_layer("output", "sigmoid", 1)
	network.fully_connect()

	data := [][][]float64{
		{{0, 0}, {0}}, // AND: 0,0 -> 0
		{{0, 1}, {0}}, // AND: 0,1 -> 0
		{{1, 0}, {0}}, // AND: 1,0 -> 0
		{{1, 1}, {1}}, // AND: 1,1 -> 1
	}

	organism := &Organism{}
	organism.init(network, 16)

	network = organism.evolve(data, 100000, 0.0001, 1000)

	// Test accuracy
	correct := 0
	for _, sample := range data {
		actual := network.run(sample[0])
		expected := sample[1][0]
		if (actual[0] >= 0.5 && expected >= 0.5) || (actual[0] < 0.5 && expected < 0.5) {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(data))
	if accuracy < 0.99 {
		t.Errorf("AND gate accuracy too low: %.2f (expected >= 99%%)", accuracy)
	}

	fmt.Printf("AND Gate Test - Accuracy: %.2f\n", accuracy)
}

// TestORGate tests evolution on OR gate logic
func TestORGate(t *testing.T) {
	network := &Network{}
	network.add_layer("input", "sigmoid", 2)
	network.add_layer("hidden", "sigmoid", 1)
	network.add_layer("output", "sigmoid", 1)
	network.fully_connect()

	data := [][][]float64{
		{{0, 0}, {0}}, // OR: 0,0 -> 0
		{{0, 1}, {1}}, // OR: 0,1 -> 1
		{{1, 0}, {1}}, // OR: 1,0 -> 1
		{{1, 1}, {1}}, // OR: 1,1 -> 1
	}

	organism := &Organism{}
	organism.init(network, 16)

	network = organism.evolve(data, 100000, 0.0001, 1000)

	// Test accuracy
	correct := 0
	for _, sample := range data {
		actual := network.run(sample[0])
		expected := sample[1][0]
		if (actual[0] >= 0.5 && expected >= 0.5) || (actual[0] < 0.5 && expected < 0.5) {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(data))
	if accuracy < 0.99 {
		t.Errorf("OR gate accuracy too low: %.2f (expected >= 99%%)", accuracy)
	}

	fmt.Printf("OR Gate Test - Accuracy: %.2f\n", accuracy)
}

// TestLinearRegression tests basic linear regression y = 2*x + 1
func TestLinearRegression(t *testing.T) {
	network := &Network{}
	network.add_layer("input", "relu", 1)    // 1 input
	network.add_layer("hidden", "relu", 3)   // 3 hidden neurons
	network.add_layer("output", "linear", 1) // 1 output, linear activation
	network.fully_connect()

	// Generate training data: y = 2*x + 1, x in [-2, 2]
	data := make([][][]float64, 20)
	for i := 0; i < 20; i++ {
		x := -2.0 + float64(i)*0.2                // -2 to 2 in steps of 0.2
		y := 2.0*x + 1.0 + (randFloat(-0.1, 0.1)) // add small noise
		data[i] = [][]float64{{x}, {y}}
	}

	organism := &Organism{}
	organism.init(network, 16)

	trainedNetwork := organism.evolve(data, 50000, 0.01, 5000)

	// Test predictions on new data points
	testPoints := []float64{-1.5, -0.5, 0.5, 1.5}
	totalError := 0.0

	for _, x := range testPoints {
		actual := trainedNetwork.run([]float64{x})[0]
		expected := 2.0*x + 1.0
		error := (actual - expected) * (actual - expected)
		totalError += error
	}

	avgError := totalError / float64(len(testPoints))
	if avgError > 0.5 { // Allow some error for noise and approximation
		t.Errorf("Linear regression average error too high: %.4f (expected <= 0.5)", avgError)
	}

	fmt.Printf("Linear Regression Test - Average MSE: %.4f\n", avgError)
}

// TestNetworkComponents tests individual network components
func TestNetworkComponents(t *testing.T) {
	// Test neuron activation
	neuron := &Neuron{
		function:   "sigmoid",
		activation: 0.0,
		bias:       0.0,
		synapses:   []Synapse{{weight: 1.0, index: 0}},
	}

	parentLayer := &Layer{
		neurons: []Neuron{{activation: 1.0}}, // input value of 1.0
	}

	neuron.activate(parentLayer)
	// sigmoid(1.0) = 1 / (1 + e^-1) ≈ 0.731058
	expected := 0.731058 // approximate value
	if neuron.activation < expected-0.01 || neuron.activation > expected+0.01 {
		t.Errorf("Sigmoid activation wrong: got %.6f, expected %.6f", neuron.activation, expected)
	}

	// Test network forward pass
	network := &Network{}
	network.add_layer("input", "sigmoid", 2)
	network.add_layer("output", "sigmoid", 1)
	network.fully_connect()

	output := network.run([]float64{1.0, 1.0})
	if len(output) != 1 {
		t.Errorf("Network output length wrong: got %d, expected 1", len(output))
	}

	fmt.Printf("Network Components Test - Sigmoid activation: %.6f\n", neuron.activation)
}
