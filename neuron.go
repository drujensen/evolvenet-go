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

type ConceptNeuron struct {
	Neuron
	concept_id        string               // unique identifier for this concept
	label             string               // human-readable name (optional)
	embedding         []float64            // distributed representation vector
	frequency         int                  // usage frequency for importance weighting
	context           []string             // common contextual concepts
	sensory_grounding map[string][]float64 // links to sensory experiences by modality
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

// ConceptNeuron methods
func (self *ConceptNeuron) clone(other *ConceptNeuron) {
	self.Neuron.clone(&other.Neuron)
	self.concept_id = other.concept_id
	self.label = other.label
	self.embedding = make([]float64, len(other.embedding))
	copy(self.embedding, other.embedding)
	self.frequency = other.frequency
	self.context = make([]string, len(other.context))
	copy(self.context, other.context)
	if other.sensory_grounding != nil {
		self.sensory_grounding = make(map[string][]float64)
		for modality, features := range other.sensory_grounding {
			self.sensory_grounding[modality] = make([]float64, len(features))
			copy(self.sensory_grounding[modality], features)
		}
	}
}

func (self *ConceptNeuron) randomize() {
	self.Neuron.randomize()
	// Initialize embedding with random values
	embedding_dim := 50 // configurable embedding dimension
	self.embedding = make([]float64, embedding_dim)
	for i := range self.embedding {
		self.embedding[i] = randFloat(-1.0, 1.0)
	}
	// Initialize sensory grounding map
	self.sensory_grounding = make(map[string][]float64)
}

func (self *ConceptNeuron) get_concept_id() string {
	return self.concept_id
}

func (self *ConceptNeuron) get_label() string {
	return self.label
}

func (self *ConceptNeuron) update_frequency() {
	self.frequency++
}

func (self *ConceptNeuron) add_sensory_grounding(modality string, features []float64) {
	if self.sensory_grounding == nil {
		self.sensory_grounding = make(map[string][]float64)
	}
	self.sensory_grounding[modality] = make([]float64, len(features))
	copy(self.sensory_grounding[modality], features)
}

func (self *ConceptNeuron) get_sensory_grounding(modality string) []float64 {
	if self.sensory_grounding == nil {
		return nil
	}
	return self.sensory_grounding[modality]
}
