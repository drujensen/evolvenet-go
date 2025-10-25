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
	neurons          []ConceptNeuron
	attention_matrix [][]float64 // attention weights between neurons
	context_window   int         // how many previous concepts to consider
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

// activate performs attention-based activation for semantic concept relationships
func (self *SemanticLayer) activate(previous_concepts []string) {
	// For each concept neuron, compute attention scores based on context
	for i := 0; i < len(self.neurons); i++ {
		neuron := &self.neurons[i]

		// Start with bias activation
		attention_sum := neuron.bias

		// Compute attention to previous context concepts
		for j, prev_concept := range previous_concepts {
			if j >= self.context_window {
				break
			}

			// Find the neuron for this previous concept
			for k, other_neuron := range self.neurons {
				if other_neuron.concept_id == prev_concept {
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

// =============================================================================
// SENSORY PROCESSING STRUCTURES
// =============================================================================

// SensoryLayer processes raw sensory inputs into feature representations
type SensoryLayer struct {
	modality        string      // "vision", "audio", "olfactory", "gustatory", "tactile"
	raw_data        []float64   // Raw sensor readings
	feature_maps    [][]float64 // Extracted features at different scales/resolutions
	resolution      []int       // Spatial/temporal dimensions of the input
	temporal_buffer [][]float64 // For temporal processing (audio, time-series)
}

// PerceptualLayer extracts modality-specific patterns and regularities
type PerceptualLayer struct {
	modality         string
	features         []FeatureNeuron // Specialized neurons for this modality
	attention_map    [][]float64     // Salience/attention weights
	temporal_context []float64       // Recent temporal context
}

// FeatureNeuron represents a learned feature detector for a specific modality
type FeatureNeuron struct {
	Neuron
	feature_type    string      // "edge", "color", "texture", "frequency", etc.
	modality        string      // Which sensory modality this responds to
	receptive_field []int       // Spatial/temporal receptive field size
	weights         [][]float64 // Feature detection weights
}

// IntegrationLayer combines information from multiple sensory modalities
type IntegrationLayer struct {
	sensory_inputs   map[string][]float64 // Current inputs from all modalities
	binding_vectors  [][]float64          // How modalities are bound together
	concept_seeds    []ConceptSeed        // Emerging concept representations
	temporal_binding []float64            // Cross-modal temporal synchronization
}

// ConceptSeed represents an emerging concept before it becomes a full ConceptNeuron
type ConceptSeed struct {
	modal_features map[string][]float64 // Features from each modality
	coherence      float64              // How well features cohere across modalities
	stability      float64              // How stable this pattern is over time
	frequency      int                  // How often this pattern occurs
}

// =============================================================================
// SENSORY LAYER METHODS
// =============================================================================

func NewSensoryLayer(modality string, dimensions []int) *SensoryLayer {
	total_size := 1
	for _, dim := range dimensions {
		total_size *= dim
	}

	layer := &SensoryLayer{
		modality:   modality,
		raw_data:   make([]float64, total_size),
		resolution: make([]int, len(dimensions)),
	}
	copy(layer.resolution, dimensions)

	return layer
}

func (self *SensoryLayer) process_input(input []float64) {
	copy(self.raw_data, input)
	// Basic preprocessing - normalize and extract simple features
	self.extract_features()
}

func (self *SensoryLayer) extract_features() {
	switch self.modality {
	case "vision":
		self.extract_visual_features()
	case "audio":
		self.extract_audio_features()
	case "olfactory":
		self.extract_olfactory_features()
	case "gustatory":
		self.extract_gustatory_features()
	case "tactile":
		self.extract_tactile_features()
	}
}

func (self *SensoryLayer) extract_visual_features() {
	// Simple edge detection and color features
	width, height := self.resolution[0], self.resolution[1]

	// Edge detection (sobel-like)
	edges := make([]float64, len(self.raw_data))
	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			idx := y*width + x
			// Simple gradient magnitude
			dx := self.raw_data[idx+1] - self.raw_data[idx-1]
			dy := self.raw_data[idx+width] - self.raw_data[idx-width]
			edges[idx] = math.Sqrt(dx*dx + dy*dy)
		}
	}

	// Color/brightness features (simplified)
	brightness := make([]float64, len(self.raw_data))
	for i, val := range self.raw_data {
		brightness[i] = val // Assume grayscale for simplicity
	}

	self.feature_maps = [][]float64{edges, brightness}
}

func (self *SensoryLayer) extract_audio_features() {
	// Simple frequency analysis using DFT-like approach
	samples := len(self.raw_data)
	features := make([]float64, samples/2)

	// Very basic frequency decomposition (simplified DFT)
	for k := 0; k < samples/2; k++ {
		var real, imag float64
		for n := 0; n < samples; n++ {
			angle := -2.0 * math.Pi * float64(k*n) / float64(samples)
			real += self.raw_data[n] * math.Cos(angle)
			imag += self.raw_data[n] * math.Sin(angle)
		}
		features[k] = math.Sqrt(real*real + imag*imag)
	}

	self.feature_maps = [][]float64{features}
}

func (self *SensoryLayer) extract_olfactory_features() {
	// Chemical pattern recognition
	patterns := make([]float64, len(self.raw_data)/4) // Group receptors

	for i := 0; i < len(patterns); i++ {
		sum := 0.0
		for j := 0; j < 4; j++ {
			sum += self.raw_data[i*4+j]
		}
		patterns[i] = sum / 4.0
	}

	self.feature_maps = [][]float64{patterns}
}

func (self *SensoryLayer) extract_gustatory_features() {
	// Taste receptor patterns
	bitter := self.raw_data[0] // Simplified single receptors
	sour := self.raw_data[1]
	salt := self.raw_data[2]
	sweet := self.raw_data[3]
	umami := self.raw_data[4]

	self.feature_maps = [][]float64{[]float64{bitter, sour, salt, sweet, umami}}
}

func (self *SensoryLayer) extract_tactile_features() {
	// Pressure and texture features
	pressure := make([]float64, len(self.raw_data))
	texture := make([]float64, len(self.raw_data)-1)

	copy(pressure, self.raw_data)

	// Simple texture as pressure gradients
	for i := 0; i < len(texture); i++ {
		texture[i] = math.Abs(self.raw_data[i+1] - self.raw_data[i])
	}

	self.feature_maps = [][]float64{pressure, texture}
}

func (self *SensoryLayer) get_features() [][]float64 {
	return self.feature_maps
}

// =============================================================================
// INTEGRATION LAYER METHODS
// =============================================================================

func NewIntegrationLayer() *IntegrationLayer {
	return &IntegrationLayer{
		sensory_inputs:  make(map[string][]float64),
		binding_vectors: make([][]float64, 0),
		concept_seeds:   make([]ConceptSeed, 0),
	}
}

func (self *IntegrationLayer) add_sensory_input(modality string, features []float64) {
	self.sensory_inputs[modality] = make([]float64, len(features))
	copy(self.sensory_inputs[modality], features)
}

func (self *IntegrationLayer) find_similar_patterns() []ConceptSeed {
	seeds := make([]ConceptSeed, 0)

	// Look for co-occurring patterns across modalities
	// This is a simplified version - real implementation would use more sophisticated clustering

	if vision, has_vision := self.sensory_inputs["vision"]; has_vision {
		if audio, has_audio := self.sensory_inputs["audio"]; has_audio {
			// Check if visual and audio patterns correlate
			correlation := self.compute_cross_modal_correlation(vision, audio)
			if correlation > 0.3 { // Threshold for pattern similarity
				seed := ConceptSeed{
					modal_features: make(map[string][]float64),
					coherence:      correlation,
					stability:      0.5,
					frequency:      1,
				}
				seed.modal_features["vision"] = make([]float64, len(vision))
				copy(seed.modal_features["vision"], vision)
				seed.modal_features["audio"] = make([]float64, len(audio))
				copy(seed.modal_features["audio"], audio)

				seeds = append(seeds, seed)
			}
		}
	}

	self.concept_seeds = seeds
	return seeds
}

func (self *IntegrationLayer) compute_cross_modal_correlation(a, b []float64) float64 {
	if len(a) != len(b) {
		// Pad shorter array
		if len(a) < len(b) {
			padded := make([]float64, len(b))
			copy(padded, a)
			a = padded
		} else {
			padded := make([]float64, len(a))
			copy(padded, b)
			b = padded
		}
	}

	return cosine_similarity(a, b)
}

func (self *IntegrationLayer) get_concept_seeds() []ConceptSeed {
	return self.concept_seeds
}

// SemanticLayer methods
func NewSemanticLayer(vocab_size int, embedding_dim int, context_window int) *SemanticLayer {
	layer := &SemanticLayer{
		layer_type:       "semantic",
		function:         "sigmoid",
		neurons:          make([]ConceptNeuron, 0, vocab_size), // Start empty, grow as needed
		attention_matrix: make([][]float64, 0, vocab_size),     // Start empty, grow as needed
		context_window:   context_window,
	}

	return layer
}

func (self *SemanticLayer) add_concept(concept_id string, label string, embedding []float64) int {
	// Check if concept already exists
	for i, neuron := range self.neurons {
		if neuron.concept_id == concept_id {
			neuron.update_frequency()
			return i
		}
	}

	// Add new concept neuron
	neuron := ConceptNeuron{
		concept_id: concept_id,
		label:      label,
		embedding:  make([]float64, len(embedding)),
		frequency:  1,
		context:    []string{},
	}
	copy(neuron.embedding, embedding)
	neuron.randomize() // initialize synapses

	// Resize attention matrix to accommodate new concept
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

func (self *SemanticLayer) get_next_concept(context []string) string {
	self.activate(context)

	// Find the most activated concept that's not in recent context
	max_activation := -1.0
	best_concept := ""

	for _, neuron := range self.neurons {
		// Skip concepts that are already in context
		in_context := false
		for _, ctx_concept := range context {
			if neuron.concept_id == ctx_concept {
				in_context = true
				break
			}
		}

		if !in_context && neuron.activation > max_activation {
			max_activation = neuron.activation
			best_concept = neuron.concept_id
		}
	}

	return best_concept
}
