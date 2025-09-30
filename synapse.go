package main

type Synapse struct {
	weight float64
	index  int
}

func (self *Synapse) clone(other *Synapse) {
	self.weight = other.weight
	self.index = other.index
}

func (self *Synapse) randomize() {
	self.weight = randFloat(-1.0, 1.0)
}

func (self *Synapse) mutate(rate float64) {
	self.weight += randFloat(-rate, rate)
}

func (self *Synapse) punctuate(pos int) {
	self.weight = roundFloat(self.weight, pos)
}
