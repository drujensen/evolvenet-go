package main

import (
	"math"
)

type Neuron struct {
	function   string
	activation float64
	bias       float64
	synapses   []Synapse
}

type WordNeuron struct {
	Neuron
	word      string    // the word/token this neuron represents
	embedding []float64 // word embedding vector
	frequency int       // usage frequency for importance weighting
	context   []string  // common contextual words
}

func (self *Neuron) clone(other *Neuron) {
	self.function = other.function
	self.activation = other.activation
	self.bias = other.bias

	if len(self.synapses) != len(other.synapses) {
		self.synapses = make([]Synapse, len(other.synapses))
	}

	for i := 0; i < len(self.synapses); i++ {
		self.synapses[i].clone(&other.synapses[i])
	}
}

func (self *Neuron) randomize() {
	self.bias = randFloat(-1.0, 1.0)
	for i := 0; i < len(self.synapses); i++ {
		self.synapses[i].randomize()
	}
}

func (self *Neuron) mutate(rate float64) {
	self.bias += randFloat(-rate, rate)
	synapse_rate := rate / float64(len(self.synapses))
	for i := 0; i < len(self.synapses); i++ {
		self.synapses[i].mutate(synapse_rate)
	}
}

func (self *Neuron) punctuate(pos int) {
	self.bias = roundFloat(self.bias, pos)
	for i := 0; i < len(self.synapses); i++ {
		self.synapses[i].punctuate(pos)
	}
}

func (self *Neuron) set_activation(value float64) {
	self.activation = value
}

func (self *Neuron) get_activation() float64 {
	return self.activation
}

func (self *Neuron) activate(parent *Layer) {
	sum := 0.0
	for i := 0; i < len(self.synapses); i++ {
		index := self.synapses[i].index
		sum += parent.neurons[index].activation * self.synapses[i].weight
	}
	sum += self.bias
	switch self.function {
	case "sigmoid":
		self.activation = 1.0 / (1.0 + math.Pow(math.E, -sum))
	case "relu":
		self.activation = math.Max(0.0, sum)
	case "tanh":
		self.activation = math.Tanh(sum)
	default:
		self.activation = sum
	}
}

// WordNeuron methods
func (self *WordNeuron) clone(other *WordNeuron) {
	self.Neuron.clone(&other.Neuron)
	self.word = other.word
	self.embedding = make([]float64, len(other.embedding))
	copy(self.embedding, other.embedding)
	self.frequency = other.frequency
	self.context = make([]string, len(other.context))
	copy(self.context, other.context)
}

func (self *WordNeuron) randomize() {
	self.Neuron.randomize()
	// Initialize embedding with random values
	embedding_dim := 50 // configurable embedding dimension
	self.embedding = make([]float64, embedding_dim)
	for i := range self.embedding {
		self.embedding[i] = randFloat(-1.0, 1.0)
	}
}

func (self *WordNeuron) get_word() string {
	return self.word
}

func (self *WordNeuron) update_frequency() {
	self.frequency++
}
