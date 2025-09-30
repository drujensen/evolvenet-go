package main

import (
	"math"
	"strings"
)

type Network struct {
	loss   float64
	layers []Layer
}

func (self *Network) add_layer(layer_type string, function string, size int) {
	layer := Layer{strings.Clone(layer_type), strings.Clone(function), make([]Neuron, size)}
	self.layers = append(self.layers, layer)
}

func (self *Network) fully_connect() {
	for i := 0; i < len(self.layers); i++ {
		if self.layers[i].layer_type == "hidden" || self.layers[i].layer_type == "output" {
			for j := 0; j < len(self.layers[i].neurons); j++ {
				self.layers[i].neurons[j].function = strings.Clone(self.layers[i].function)
				for k := 0; k < len(self.layers[i-1].neurons); k++ {
					self.layers[i].neurons[j].synapses = append(self.layers[i].neurons[j].synapses, Synapse{0.5, k})
				}
			}
		}
	}
}

func (self *Network) clone(other *Network) {
	self.loss = other.loss

	if len(self.layers) != len(other.layers) {
		self.layers = make([]Layer, len(other.layers))
	}

	for i := 0; i < len(self.layers); i++ {
		self.layers[i].clone(&other.layers[i])
	}
}

func (self *Network) randomize() {
	self.loss = 1.0
	for i := 0; i < len(self.layers); i++ {
		self.layers[i].randomize()
	}
}

func (self *Network) mutate() {
	for i := 0; i < len(self.layers); i++ {
		self.layers[i].mutate(0.1)
	}
}

func (self *Network) punctuate(pos int) {
	for i := 0; i < len(self.layers); i++ {
		self.layers[i].punctuate(pos)
	}
}

func (self *Network) run(data []float64) []float64 {
	for i := 0; i < len(self.layers); i++ {
		if i == 0 {
			self.layers[i].set_activations(data)
		} else {
			self.layers[i].activate(&self.layers[i-1])
		}
	}

	return self.layers[len(self.layers)-1].get_activations()
}

func (self *Network) evaluate(data [][][]float64) {
	sum := 0.0
	for i := 0; i < len(data); i++ {
		actual := self.run(data[i][0])
		expected := data[i][1]
		for j := 0; j < len(expected); j++ {
			sum += math.Pow(expected[j]-actual[j], 2)
		}
	}
	self.loss = float64(sum / (2.0 * float64(len(data))))
}

func (self *Network) get_loss() float64 {
	return self.loss
}
