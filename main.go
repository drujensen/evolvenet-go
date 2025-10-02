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

func avg(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func avgInt(s []int) float64 {
	sum := 0
	for _, v := range s {
		sum += v
	}
	return float64(sum) / float64(len(s))
}

func main() {
	const numTrials = 10

	var networkCMA, networkGA *Network

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

	// Benchmark CMA-ES
	var cmaGens []int
	var cmaLosses []float64
	var cmaAccs []float64

	fmt.Println("Benchmarking CMA-ES (10 trials)...")
	for t := 0; t < numTrials; t++ {
		organism.init(network, 16)
		net, gen, loss := organism.evolveCMA(data, 100000, 0.000001, 1000, network)
		cmaGens = append(cmaGens, gen)
		cmaLosses = append(cmaLosses, loss)

		correct := 0
		for _, sample := range data {
			actual := net.run(sample[0])
			if (actual[0] >= 0.5 && sample[1][0] >= 0.5) || (actual[0] < 0.5 && sample[1][0] < 0.5) {
				correct++
			}
		}
		acc := float64(correct) / float64(len(data))
		cmaAccs = append(cmaAccs, acc)
		fmt.Printf("CMA-ES Trial %d: gen=%d, loss=%f, acc=%f\n", t+1, gen, loss, acc)
		networkCMA = net // Save last
	}

	// Benchmark GA
	var gaGens []int
	var gaLosses []float64
	var gaAccs []float64

	fmt.Println("Benchmarking Momentum-GA (10 trials)...")
	for t := 0; t < numTrials; t++ {
		organism.init(network, 16)
		net, gen, loss := organism.evolveGA(data, 100000, 0.000001, 1000)
		gaGens = append(gaGens, gen)
		gaLosses = append(gaLosses, loss)

		correct := 0
		for _, sample := range data {
			actual := net.run(sample[0])
			if (actual[0] >= 0.5 && sample[1][0] >= 0.5) || (actual[0] < 0.5 && sample[1][0] < 0.5) {
				correct++
			}
		}
		acc := float64(correct) / float64(len(data))
		gaAccs = append(gaAccs, acc)
		fmt.Printf("GA Trial %d: gen=%d, loss=%f, acc=%f\n", t+1, gen, loss, acc)
		networkGA = net // Save last
	}

	// Print averages
	fmt.Printf("CMA-ES averages: generations=%.0f, loss=%.6f, accuracy=%.4f\n", avgInt(cmaGens), avg(cmaLosses), avg(cmaAccs))
	fmt.Printf("Momentum-GA averages: generations=%.0f, loss=%.6f, accuracy=%.4f\n", avgInt(gaGens), avg(gaLosses), avg(gaAccs))

	// Individual final tests (using last run networks)
	tn, tp, fn, fp, ct := 0, 0, 0, 0, 0
	for i := 0; i < len(data); i++ {
		actual := networkCMA.run(data[i][0])
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
	fmt.Printf("Test size: %d\n----------------------\nTN: %d | FP: %d\n----------------------\nFN: %d | TP: %d\n----------------------\nAccuracy: %f\n", len(data), tn, fp, fn, tp, float64(tn+tp)/float64(ct))

	fmt.Println("\nFinal GA results:")
	testNetwork(networkGA, data, "GA")
	tn, tp, fn, fp, ct = 0, 0, 0, 0, 0
	for i := 0; i < len(data); i++ {
		actual := networkGA.run(data[i][0])
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
	fmt.Printf("Test size: %d\n----------------------\nTN: %d | FP: %d\n----------------------\nFN: %d | TP: %d\n----------------------\nAccuracy: %f\n", len(data), tn, fp, fn, tp, float64(tn+tp)/float64(ct))
}

/*\n   fmt.Printf(\"generation: %d loss: %f. below threshold. breaking\\n\", i, loss)\n   fmt.Printf(\"layer 0, neuron 0\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[0].neurons[0].activation)\n   fmt.Printf(\"layer 0, neuron 1\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[0].neurons[1].activation)\n   fmt.Printf(\"layer 1, neuron 0\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[1].neurons[0].activation)\n   fmt.Printf(\"layer function: %s\\n\", self.networks[0].layers[1].function)\n   fmt.Printf(\"neuron function: %s\\n\", self.networks[0].layers[1].neurons[0].function)\n   fmt.Printf(\"weight1: %f\\n\", self.networks[0].layers[1].neurons[0].synapses[0].weight)\n   fmt.Printf(\"weight2: %f\\n\", self.networks[0].layers[1].neurons[0].synapses[1].weight)\n   fmt.Printf(\"bias: %f\\n\", self.networks[0].layers[1].neurons[0].bias)\n   fmt.Printf(\"layer 1, neuron 1\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[1].neurons[1].activation)\n   fmt.Printf(\"function: %s\\n\", self.networks[0].layers[1].neurons[1].function)\n   fmt.Printf(\"weight1: %f\\n\", self.networks[0].layers[1].neurons[1].synapses[0].weight)\n   fmt.Printf(\"weight2: %f\\n\", self.networks[0].layers[1].neurons[1].synapses[1].weight)\n   fmt.Printf(\"bias: %f\\n\", self.networks[0].layers[1].neurons[1].bias)\n   fmt.Printf(\"layer 2, neuron 0\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[2].neurons[0].activation)\n   fmt.Printf(\"function: %s\\n\", self.networks[0].layers[2].neurons[0].function)\n   fmt.Printf(\"weight1: %f\\n\", self.networks[0].layers[2].neurons[0].synapses[0].weight)\n   fmt.Printf(\"weight2: %f\\n\", self.networks[0].layers[2].neurons[0].synapses[1].weight)\n   fmt.Printf(\"bias: %f\\n\", self.networks[0].layers[2].neurons[0].bias)\n*/
