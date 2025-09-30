package main

type Layer struct {
	layer_type string
	function   string
	neurons    []Neuron
}

func (self *Layer) clone(other *Layer) {
	self.layer_type = other.layer_type
	self.function = other.function

	if len(self.neurons) != len(other.neurons) {
		self.neurons = make([]Neuron, len(other.neurons))
	}

	for i := 0; i < len(self.neurons); i++ {
		self.neurons[i].clone(&other.neurons[i])
	}
}

func (self *Layer) randomize() {
	for i := 0; i < len(self.neurons); i++ {
		self.neurons[i].randomize()
	}
}

func (self *Layer) mutate(rate float64) {
	neuron_rate := rate / float64(len(self.neurons))
	for i := 0; i < len(self.neurons); i++ {
		self.neurons[i].mutate(neuron_rate)
	}
}

func (self *Layer) punctuate(pos int) {
	for i := 0; i < len(self.neurons); i++ {
		self.neurons[i].punctuate(pos)
	}
}

func (self *Layer) set_activations(data []float64) {
	for i := 0; i < len(self.neurons); i++ {
		self.neurons[i].set_activation(data[i])
	}
}

func (self *Layer) get_activations() []float64 {
	data := make([]float64, len(self.neurons))
	for i := 0; i < len(self.neurons); i++ {
		data[i] = self.neurons[i].get_activation()
	}
	return data
}

func (self *Layer) activate(parent *Layer) {
	for i := 0; i < len(self.neurons); i++ {
		self.neurons[i].activate(parent)
	}
}
