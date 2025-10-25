package main

import (
	"fmt"
	"sort"
)

type Organism struct {
	networks []Network
}

func (self *Organism) init(network *Network, size int) {
	self.networks = make([]Network, size)
	for i := 0; i < size; i++ {
		self.networks[i].clone(network)
		self.networks[i].randomize()
	}
}

func (self *Organism) evolve(data [][][]float64, generations int, threshold float64, log_every int) *Network {
	network, _, _ := self.evolveGA(data, generations, threshold, log_every)
	return network
}

func (self *Organism) evolveGA(data [][][]float64, generations int, threshold float64, log_every int) (*Network, int, float64) {
	half := len(self.networks) / 2
	quarter := half / 2

	gen := generations
	// for every generation
	for i := 0; i < generations; i++ {

		// evaluate the networks
		for j := 0; j < len(self.networks); j++ {
			self.networks[j].evaluate(data)
		}

		// sort the networks by loss
		sort.Slice(self.networks, func(i, j int) bool { return self.networks[i].get_loss() < self.networks[j].get_loss() })

		loss := self.networks[0].get_loss()
		if threshold > 0 && loss < threshold {
			gen = i
			break
		}

		if i%log_every == 0 {
			fmt.Printf("generation: %d loss: %f\n", i, loss)
		}

		// clone the top quarter to the bottom quarter of the networks
		for j := 0; j < quarter; j++ {
			self.networks[half+quarter+j].clone(&self.networks[j])
		}

		// punctuate several of the networks
		for j := 1; j < quarter; j++ {
			self.networks[j].punctuate(j)
		}

		// mutate all but the best and punctuated networks
		for j := quarter; j < len(self.networks); j++ {
			self.networks[j].mutate()
		}
	}
	// sort the networks by fitness
	sort.Slice(self.networks, func(i, j int) bool { return self.networks[i].get_loss() < self.networks[j].get_loss() })
	return &self.networks[0], gen, self.networks[0].get_loss()
}
