package main

import (
	"fmt"
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func testNetwork(network *Network, data [][][]float64, label string) {
	correct := 0
	for _, sample := range data {
		actual := network.run(sample[0])
		expected := sample[1][0]
		if (actual[0] >= 0.5 && expected >= 0.5) || (actual[0] < 0.5 && expected < 0.5) {
			correct++
		}
	}
	fmt.Printf("testNetwork(***network.Network): %d / %d correct\n", correct, len(data))
}

func main() {
	network := &Network{}
	network.add_layer("input", "sigmoid", 2)
	network.add_layer("hidden", "sigmoid", 2)
	network.add_layer("output", "sigmoid", 1)
	network.fully_connect()

	data := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	organism := &Organism{}
	organism.init(network, 16)

	fmt.Println("Testing CMA-ES strategy:")
	startCMA := time.Now()
	networkCMA := organism.evolveCMA(data, 100000, 0.1, 1000, network)
	durationCMA := time.Since(startCMA)
	fmt.Printf("CMA-ES time: %v\n", durationCMA)

	// Reset organism for GA
	organism.init(network, 16)
	fmt.Println("Testing GA strategy:")
	startGA := time.Now()
	networkGA := organism.evolveGA(data, 100000, 0.000001, 1000)
	durationGA := time.Since(startGA)
	fmt.Printf("GA time: %v\n", durationGA)

	// Test accuracies
	fmt.Println("\nCMA-ES results:")
	testNetwork(networkCMA, data, "CMA-ES")

	fmt.Println("\nGA results:")
	testNetwork(networkGA, data, "GA")

	// Use CMA-ES for final network
	network = networkCMA
	tn, tp, fn, fp, ct := 0, 0, 0, 0, 0

	for i := 0; i < len(data); i++ {
		actual := network.run(data[i][0])
		expected := data[i][1]
		for j := 0; j < len(expected); j++ {
			ct++
			if expected[j] > 0.5 {
				if actual[j] > 0.5 {
					tp += 1
				} else {
					fp += 1
				}
			} else {
				if actual[j] <= 0.5 {
					tn += 1
				} else {
					fn += 1
				}
			}
		}
	}

	fmt.Printf("Test size: %d\n", len(data))
	fmt.Printf("----------------------\n")
	fmt.Printf("TN: %d | FP: %d\n", tn, fp)
	fmt.Printf("----------------------\n")
	fmt.Printf("FN: %d | TP: %d\n", fn, tp)
	fmt.Printf("----------------------\n")
	fmt.Printf("Accuracy: %f\n", float64(tn+tp)/float64(ct))
}

/*
   fmt.Printf("generation: %d loss: %f. below threshold. breaking\n", i, loss)
   fmt.Printf("layer 0, neuron 0\n",)
   fmt.Printf("activation: %f\n", self.networks[0].layers[0].neurons[0].activation)
   fmt.Printf("layer 0, neuron 1\n",)
   fmt.Printf("activation: %f\n", self.networks[0].layers[0].neurons[1].activation)
   fmt.Printf("layer 1, neuron 0\n",)
   fmt.Printf("activation: %f\n", self.networks[0].layers[1].neurons[0].activation)
   fmt.Printf("layer function: %s\n", self.networks[0].layers[1].function)
   fmt.Printf("neuron function: %s\n", self.networks[0].layers[1].neurons[0].function)
   fmt.Printf("weight1: %f\n", self.networks[0].layers[1].neurons[0].synapses[0].weight)
   fmt.Printf("weight2: %f\n", self.networks[0].layers[1].neurons[0].synapses[1].weight)
   fmt.Printf("bias: %f\n", self.networks[0].layers[1].neurons[0].bias)
   fmt.Printf("layer 1, neuron 1\n",)
   fmt.Printf("activation: %f\n", self.networks[0].layers[1].neurons[1].activation)
   fmt.Printf("function: %s\n", self.networks[0].layers[1].neurons[1].function)
   fmt.Printf("weight1: %f\n", self.networks[0].layers[1].neurons[1].synapses[0].weight)
   fmt.Printf("weight2: %f\n", self.networks[0].layers[1].neurons[1].synapses[1].weight)
   fmt.Printf("bias: %f\n", self.networks[0].layers[1].neurons[1].bias)
   fmt.Printf("layer 2, neuron 0\n",)
   fmt.Printf("activation: %f\n", self.networks[0].layers[2].neurons[0].activation)
   fmt.Printf("function: %s\n", self.networks[0].layers[2].neurons[0].function)
   fmt.Printf("weight1: %f\n", self.networks[0].layers[2].neurons[0].synapses[0].weight)
   fmt.Printf("weight2: %f\n", self.networks[0].layers[2].neurons[0].synapses[1].weight)
   fmt.Printf("bias: %f\n", self.networks[0].layers[2].neurons[0].bias)
*/
