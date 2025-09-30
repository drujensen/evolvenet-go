package main

import (
	"math"
	"math/rand"
)

func randFloat(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func roundFloat(f float64, n int) float64 {
	return math.Round(f*math.Pow10(n)) / math.Pow10(n)
}

// flattenNetwork extracts all trainable parameters (biases and synapse weights) from the network into a single slice
func flattenNetwork(net *Network) []float64 {
	var params []float64
	for i := range net.layers {
		for j := range net.layers[i].neurons {
			params = append(params, net.layers[i].neurons[j].bias)
			for k := range net.layers[i].neurons[j].synapses {
				params = append(params, net.layers[i].neurons[j].synapses[k].weight)
			}
		}
	}
	return params
}

// randomizeNetworkWeights sets all biases and synapse weights to random values
func randomizeNetworkWeights(net *Network) {
	for i := range net.layers {
		for j := range net.layers[i].neurons {
			net.layers[i].neurons[j].bias = randFloat(-1.0, 1.0)
			for k := range net.layers[i].neurons[j].synapses {
				net.layers[i].neurons[j].synapses[k].weight = randFloat(-1.0, 1.0)
			}
		}
	}
}

// unflattenNetwork creates a new network by copying the template structure and setting parameters from the flattened slice
func unflattenNetwork(params []float64, template *Network) *Network {
	newNet := &Network{}
	newNet.layers = make([]Layer, len(template.layers))
	idx := 0

	for i := range template.layers {
		newNet.layers[i] = Layer{
			layer_type: template.layers[i].layer_type,
			function:   template.layers[i].function,
			neurons:    make([]Neuron, len(template.layers[i].neurons)),
		}
		for j := range template.layers[i].neurons {
			newNet.layers[i].neurons[j] = Neuron{
				function: template.layers[i].neurons[j].function,
				synapses: make([]Synapse, len(template.layers[i].neurons[j].synapses)),
			}
			for k := range template.layers[i].neurons[j].synapses {
				newNet.layers[i].neurons[j].synapses[k] = Synapse{
					index: template.layers[i].neurons[j].synapses[k].index,
				}
			}
			newNet.layers[i].neurons[j].bias = params[idx]
			idx++
			for k := range template.layers[i].neurons[j].synapses {
				newNet.layers[i].neurons[j].synapses[k].weight = params[idx]
				idx++
			}
		}
	}
	return newNet
}
