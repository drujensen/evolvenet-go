package main

type Synapse struct {
	weight  float64
	index   int
	history []float64
}

func (self *Synapse) clone(other *Synapse) {
	self.weight = other.weight
	self.index = other.index
	self.history = make([]float64, len(other.history))
	copy(self.history, other.history)
}

func (self *Synapse) randomize() {
	self.weight = randFloat(-1.0, 1.0)
}

func (self *Synapse) mutate(rate float64) {
	const maxHistory = 10
	const alpha = 0.9 // Bias strength: 0.9 favors history, 0.1 randomness

	// Compute mean of historical deltas
	var meanDelta float64
	if len(self.history) > 0 {
		sum := 0.0
		for _, d := range self.history {
			sum += d
		}
		meanDelta = sum / float64(len(self.history))
	}

	// Compute biased random delta
	randomDelta := randFloat(-rate, rate)
	delta := alpha*meanDelta + (1-alpha)*randomDelta

	// Apply the delta
	self.weight += delta

	// Record the delta in history
	self.history = append(self.history, delta)
	if len(self.history) > maxHistory {
		self.history = self.history[1:] // Remove oldest
	}
}

func (self *Synapse) punctuate(pos int) {
	self.weight = roundFloat(self.weight, pos)
}
