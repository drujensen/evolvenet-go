package main

import (
	"math"
)

type Layer struct {
	layer_type string
	function   string
	neurons    []Neuron
}

type SemanticLayer struct {
	layer_type       string
	function         string
	neurons          []WordNeuron
	attention_matrix [][]float64 // attention weights between neurons
	context_window   int         // how many previous words to consider
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

// activate performs attention-based activation for semantic word relationships
func (self *SemanticLayer) activate(previous_words []string) {
	// For each word neuron, compute attention scores based on context
	for i := 0; i < len(self.neurons); i++ {
		neuron := &self.neurons[i]

		// Start with bias activation
		attention_sum := neuron.bias

		// Compute attention to previous context words
		for j, prev_word := range previous_words {
			if j >= self.context_window {
				break
			}

			// Find the neuron for this previous word
			for k, other_neuron := range self.neurons {
				if other_neuron.word == prev_word {
					// Compute attention score based on embedding similarity
					similarity := cosine_similarity(neuron.embedding, other_neuron.embedding)

					// Apply attention weight and add to activation
					attention_weight := self.attention_matrix[i][k]
					attention_sum += similarity * attention_weight * other_neuron.activation

					break
				}
			}
		}

		// Apply activation function
		switch self.function {
		case "sigmoid":
			neuron.activation = 1.0 / (1.0 + math.Pow(math.E, -attention_sum))
		case "relu":
			neuron.activation = math.Max(0.0, attention_sum)
		case "tanh":
			neuron.activation = math.Tanh(attention_sum)
		default:
			neuron.activation = attention_sum
		}
	}
}

// cosine_similarity computes cosine similarity between two vectors
func cosine_similarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dot_product, norm_a, norm_b float64
	for i := 0; i < len(a); i++ {
		dot_product += a[i] * b[i]
		norm_a += a[i] * a[i]
		norm_b += b[i] * b[i]
	}

	if norm_a == 0 || norm_b == 0 {
		return 0.0
	}

	return dot_product / (math.Sqrt(norm_a) * math.Sqrt(norm_b))
}

// SemanticLayer methods
func NewSemanticLayer(vocab_size int, embedding_dim int, context_window int) *SemanticLayer {
	layer := &SemanticLayer{
		layer_type:       "semantic",
		function:         "sigmoid",
		neurons:          make([]WordNeuron, 0, vocab_size), // Start empty, grow as needed
		attention_matrix: make([][]float64, 0, vocab_size),  // Start empty, grow as needed
		context_window:   context_window,
	}

	return layer
}

func (self *SemanticLayer) add_word(word string, embedding []float64) int {
	// Check if word already exists
	for i, neuron := range self.neurons {
		if neuron.word == word {
			neuron.update_frequency()
			return i
		}
	}

	// Add new word neuron
	neuron := WordNeuron{
		word:      word,
		embedding: make([]float64, len(embedding)),
		frequency: 1,
		context:   []string{},
	}
	copy(neuron.embedding, embedding)
	neuron.randomize() // initialize synapses

	// Resize attention matrix to accommodate new word
	new_size := len(self.neurons) + 1
	for i := range self.attention_matrix {
		// Extend existing rows
		self.attention_matrix[i] = append(self.attention_matrix[i], randFloat(0.0, 1.0))
	}
	// Add new row
	new_row := make([]float64, new_size)
	for j := range new_row {
		new_row[j] = randFloat(0.0, 1.0)
	}
	self.attention_matrix = append(self.attention_matrix, new_row)

	self.neurons = append(self.neurons, neuron)
	return len(self.neurons) - 1
}

func (self *SemanticLayer) get_next_word(context []string) string {
	self.activate(context)

	// Find the most activated word that's not in recent context
	max_activation := -1.0
	best_word := ""

	for _, neuron := range self.neurons {
		// Skip words that are already in context
		in_context := false
		for _, ctx_word := range context {
			if neuron.word == ctx_word {
				in_context = true
				break
			}
		}

		if !in_context && neuron.activation > max_activation {
			max_activation = neuron.activation
			best_word = neuron.word
		}
	}

	return best_word
}
