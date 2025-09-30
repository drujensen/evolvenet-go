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
	return self.evolveGA(data, generations, threshold, log_every)
}

func (self *Organism) evolveGA(data [][][]float64, generations int, threshold float64, log_every int) *Network {
	half := len(self.networks) / 2
	quarter := half / 2

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
	return &self.networks[0]
}

func (self *Organism) evolveCMA(data [][][]float64, generations int, threshold float64, log_every int, template *Network) *Network {
	dim := len(flattenNetwork(template)) // Get dimension from template
	lambda := len(self.networks)         // Population size
	cma := NewCMAES(dim, lambda)

	// Initialize CMA-ES mean with flattened template weights
	randomizeNetworkWeights(template)
	cma.m = flattenNetwork(template)

	bestNetwork := template
	bestLoss := 1.0

	for i := 0; i < generations; i++ {
		// Sample new population
		population := cma.sample()

		// Evaluate each candidate
		networks := make([]*Network, lambda)
		losses := make([]float64, lambda)

		for j := 0; j < lambda; j++ {
			networks[j] = unflattenNetwork(population[j], template)
			networks[j].evaluate(data)
			losses[j] = networks[j].get_loss()

			if losses[j] < bestLoss {
				bestLoss = losses[j]
				bestNetwork = networks[j]
			}
		}

		// Check threshold
		if threshold > 0 && bestLoss < threshold {
			break
		}

		if i%log_every == 0 {
			fmt.Printf("CMA-ES generation: %d best loss: %f\n", i, bestLoss)
		}

		// Select top mu (lambda/2) individuals
		type candidate struct {
			index int
			loss  float64
		}

		candidates := make([]candidate, lambda)
		for j := 0; j < lambda; j++ {
			candidates[j] = candidate{j, losses[j]}
		}

		sort.Slice(candidates, func(a, b int) bool {
			return candidates[a].loss < candidates[b].loss
		})

		// Select best mu
		selected := make([][]float64, cma.mu)
		for j := 0; j < cma.mu; j++ {
			selected[j] = population[candidates[j].index]
		}

		// Update CMA-ES parameters
		cma.update(selected)
	}

	return bestNetwork
}
