package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Example of multimodal sensory processing and concept formation
func demonstrate_multimodal_processing() {
	fmt.Println("=== Multimodal Sensory Processing Demo ===")

	// Create sensory processing layers for different modalities
	vision_layer := NewSensoryLayer("vision", []int{10, 10})   // 10x10 pixel image
	olfactory_layer := NewSensoryLayer("olfactory", []int{20}) // 20 chemical receptors
	gustatory_layer := NewSensoryLayer("gustatory", []int{5})  // 5 taste receptors
	tactile_layer := NewSensoryLayer("tactile", []int{25})     // 5x5 pressure sensors

	// Create integration layer
	integration_layer := NewIntegrationLayer()

	// Create semantic layer for concepts
	semantic_layer := NewSemanticLayer(50, 32, 3)

	// Example 1: Processing an "apple" experience
	fmt.Println("\n--- Processing APPLE Experience ---")

	// Simulate visual input (round, red shape)
	visual_input := make([]float64, 100) // 10x10 image
	for i := range visual_input {
		x, y := i%10, i/10
		// Create a circular red shape in the center
		center_x, center_y := 5.0, 5.0
		dist := math.Sqrt(math.Pow(float64(x)-center_x, 2) + math.Pow(float64(y)-center_y, 2))
		if dist < 3 {
			visual_input[i] = 0.8 // red
		} else {
			visual_input[i] = 0.2 // background
		}
	}
	vision_layer.process_input(visual_input)

	// Simulate taste input (sweet, slightly sour)
	taste_input := []float64{0.1, 0.3, 0.0, 0.8, 0.2} // bitter, sour, salt, sweet, umami
	gustatory_layer.process_input(taste_input)

	// Simulate smell input (fruity, fresh)
	smell_input := make([]float64, 20)
	for i := range smell_input {
		if i < 10 {
			smell_input[i] = 0.7 // fruity notes
		} else {
			smell_input[i] = 0.3 // fresh notes
		}
	}
	olfactory_layer.process_input(smell_input)

	// Simulate touch input (smooth, firm)
	touch_input := make([]float64, 25) // 5x5 pressure grid
	for i := range touch_input {
		touch_input[i] = 0.6 // moderate pressure, smooth texture
	}
	tactile_layer.process_input(touch_input)

	// Integrate sensory inputs
	integration_layer.add_sensory_input("vision", flatten_features(vision_layer.get_features()))
	integration_layer.add_sensory_input("gustatory", flatten_features(gustatory_layer.get_features()))
	integration_layer.add_sensory_input("olfactory", flatten_features(olfactory_layer.get_features()))
	integration_layer.add_sensory_input("tactile", flatten_features(tactile_layer.get_features()))

	// Find emerging concept patterns
	seeds := integration_layer.find_similar_patterns()

	fmt.Printf("Found %d concept seeds from multimodal integration\n", len(seeds))
	for i, seed := range seeds {
		fmt.Printf("Seed %d: coherence=%.3f, modalities=%v\n", i, seed.coherence, get_modalities(seed))
	}

	// Create a concept from this experience
	apple_embedding := create_concept_embedding(seeds)
	semantic_layer.add_concept("APPLE", "apple", apple_embedding)

	// Add sensory grounding to the concept
	apple_neuron := &semantic_layer.neurons[len(semantic_layer.neurons)-1]
	apple_neuron.add_sensory_grounding("vision", flatten_features(vision_layer.get_features()))
	apple_neuron.add_sensory_grounding("gustatory", flatten_features(gustatory_layer.get_features()))
	apple_neuron.add_sensory_grounding("olfactory", flatten_features(olfactory_layer.get_features()))
	apple_neuron.add_sensory_grounding("tactile", flatten_features(tactile_layer.get_features()))

	fmt.Println("Created APPLE concept with multimodal grounding")

	// Example 2: Text input processing
	fmt.Println("\n--- Processing Text Input: 'red apple' ---")

	// Add word concepts
	semantic_layer.add_concept("RED", "red", create_random_embedding(32))
	semantic_layer.add_concept("ROUND", "round", create_random_embedding(32))
	semantic_layer.add_concept("SWEET", "sweet", create_random_embedding(32))

	// Simulate language understanding through concept activation
	context := []string{"RED", "APPLE"}
	fmt.Printf("Text context: %v\n", context)

	next_concept := semantic_layer.get_next_concept(context)
	if next_concept != "" {
		fmt.Printf("Predicted next concept: %s\n", next_concept)
	}

	fmt.Println("=== End Multimodal Demo ===")
}

// Demonstrate how different input types could be processed
func demonstrate_input_types() {
	fmt.Println("=== Input Type Processing Examples ===")

	// 1. Text Input Processing
	fmt.Println("\n--- Text Input: Processing a sentence ---")
	text_input := "The red apple tastes sweet and smells fresh"
	words := strings.Fields(text_input)

	semantic_layer := NewSemanticLayer(100, 32, 3)

	// Add word concepts
	word_concepts := []string{"THE", "RED", "APPLE", "TASTES", "SWEET", "AND", "SMELLS", "FRESH"}
	for _, word := range word_concepts {
		embedding := create_random_embedding(32)
		semantic_layer.add_concept(strings.ToUpper(word), word, embedding)
	}

	// Process sentence as sequence
	context := []string{}
	for _, word := range words {
		concept_id := strings.ToUpper(word)
		context = append(context, concept_id)

		if len(context) > 3 {
			context = context[1:] // Keep only recent context
		}

		next := semantic_layer.get_next_concept(context)
		if next != "" {
			fmt.Printf("Input: %-8s | Context: %v | Predicts: %s\n", word, context, next)
		}
	}

	// 2. Image Input Processing
	fmt.Println("\n--- Image Input: Processing visual features ---")
	vision_layer := NewSensoryLayer("vision", []int{8, 8}) // 8x8 pixel image

	// Create a simple "face-like" pattern
	image := make([]float64, 64)
	// Eyes
	image[1*8+2] = 1.0
	image[1*8+5] = 1.0
	// Nose
	image[3*8+3] = 0.8
	image[3*8+4] = 0.8
	// Mouth
	for x := 2; x <= 5; x++ {
		image[5*8+x] = 0.6
	}

	vision_layer.process_input(image)
	features := vision_layer.get_features()

	fmt.Printf("Image features extracted: %d feature maps\n", len(features))
	fmt.Printf("Edge features: %d values\n", len(features[0]))
	fmt.Printf("Brightness features: %d values\n", len(features[1]))

	// 3. Audio Input Processing
	fmt.Println("\n--- Audio Input: Processing sound patterns ---")
	audio_layer := NewSensoryLayer("audio", []int{64}) // 64 audio samples

	// Create a simple sine wave (representing a tone)
	audio := make([]float64, 64)
	frequency := 5.0 // 5 Hz relative to sample rate
	for i := 0; i < 64; i++ {
		audio[i] = math.Sin(2 * math.Pi * frequency * float64(i) / 64.0)
	}

	audio_layer.process_input(audio)
	audio_features := audio_layer.get_features()

	fmt.Printf("Audio features extracted: %d feature maps\n", len(audio_features))
	fmt.Printf("Frequency spectrum: %d frequency bins\n", len(audio_features[0]))

	// 4. Integration Example
	fmt.Println("\n--- Integration: Combining modalities ---")
	integration_layer := NewIntegrationLayer()

	// Add features from different modalities
	integration_layer.add_sensory_input("vision", flatten_features(features))
	integration_layer.add_sensory_input("audio", flatten_features(audio_features))

	seeds := integration_layer.find_similar_patterns()
	fmt.Printf("Cross-modal integration found %d concept patterns\n", len(seeds))

	if len(seeds) > 0 {
		fmt.Printf("Pattern coherence: %.3f\n", seeds[0].coherence)
		fmt.Printf("Modalities integrated: %v\n", get_modalities(seeds[0]))
	}

	fmt.Println("=== End Input Types Demo ===")
}

// Helper functions for the demo
func flatten_features(feature_maps [][]float64) []float64 {
	total_size := 0
	for _, fmap := range feature_maps {
		total_size += len(fmap)
	}

	flattened := make([]float64, 0, total_size)
	for _, fmap := range feature_maps {
		flattened = append(flattened, fmap...)
	}

	return flattened
}

func get_modalities(seed ConceptSeed) []string {
	modalities := make([]string, 0, len(seed.modal_features))
	for modality := range seed.modal_features {
		modalities = append(modalities, modality)
	}
	return modalities
}

func create_concept_embedding(seeds []ConceptSeed) []float64 {
	if len(seeds) == 0 {
		return create_random_embedding(32)
	}

	// Simple approach: average features from different modalities
	embedding := make([]float64, 32)
	seed := seeds[0] // Use first seed for simplicity

	feature_count := 0
	for _, features := range seed.modal_features {
		for i, feature := range features {
			if i < len(embedding) {
				embedding[i] += feature
				feature_count++
			}
		}
	}

	// Normalize
	if feature_count > 0 {
		for i := range embedding {
			embedding[i] /= float64(feature_count)
		}
	}

	return embedding
}

func create_random_embedding(size int) []float64 {
	embedding := make([]float64, size)
	for i := range embedding {
		embedding[i] = randFloat(-1.0, 1.0)
	}
	return embedding
}

func testNetwork(network *Network, data [][][]float64, label string) {
	correct := 0
	for _, sample := range data {
		actual := network.run(sample[0])
		expected := sample[1][0]
		if (actual[0] >= 0.5 && expected >= 0.5) || (actual[0] < 0.5 && expected < 0.5) {
			correct++
		}
	}
	fmt.Printf("testNetwork(***network.Network): %d / %d correct\n", correct, len(data))
}

func avg(s []float64) float64 {
	sum := 0.0
	for _, v := range s {
		sum += v
	}
	return sum / float64(len(s))
}

func avgInt(s []int) float64 {
	sum := 0
	for _, v := range s {
		sum += v
	}
	return float64(sum) / float64(len(s))
}

func main() {
	const numTrials = 10

	var networkGA *Network

	network := &Network{}
	network.add_layer("input", "sigmoid", 2)
	network.add_layer("hidden", "sigmoid", 2)
	network.add_layer("output", "sigmoid", 1)
	network.fully_connect()

	data := [][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	}

	organism := &Organism{}

	// Benchmark GA
	var gaGens []int
	var gaLosses []float64
	var gaAccs []float64

	fmt.Println("Benchmarking Momentum-GA (10 trials)...")
	for t := 0; t < numTrials; t++ {
		organism.init(network, 16)
		net, gen, loss := organism.evolveGA(data, 100000, 0.000001, 1000)
		gaGens = append(gaGens, gen)
		gaLosses = append(gaLosses, loss)

		correct := 0
		for _, sample := range data {
			actual := net.run(sample[0])
			if (actual[0] >= 0.5 && sample[1][0] >= 0.5) || (actual[0] < 0.5 && sample[1][0] < 0.5) {
				correct++
			}
		}
		acc := float64(correct) / float64(len(data))
		gaAccs = append(gaAccs, acc)
		fmt.Printf("GA Trial %d: gen=%d, loss=%f, acc=%f\n", t+1, gen, loss, acc)
		networkGA = net // Save last
	}

	// Print averages
	fmt.Printf("Momentum-GA averages: generations=%.0f, loss=%.6f, accuracy=%.4f\n", avgInt(gaGens), avg(gaLosses), avg(gaAccs))

	// Final GA results
	fmt.Println("\nFinal GA results:")
	testNetwork(networkGA, data, "GA")
	tn, tp, fn, fp, ct := 0, 0, 0, 0, 0
	for i := 0; i < len(data); i++ {
		actual := networkGA.run(data[i][0])
		expected := data[i][1]
		for j := 0; j < len(expected); j++ {
			ct++
			if expected[j] > 0.5 {
				if actual[j] > 0.5 {
					tp += 1
				} else {
					fp += 1
				}
			} else {
				if actual[j] <= 0.5 {
					tn += 1
				} else {
					fn += 1
				}
			}
		}
	}
	fmt.Printf("Test size: %d\n----------------------\nTN: %d | FP: %d\n----------------------\nFN: %d | TP: %d\n----------------------\nAccuracy: %f\n", len(data), tn, fp, fn, tp, float64(tn+tp)/float64(ct))

	// Demonstrate multimodal processing
	demonstrate_multimodal_processing()

	// Demonstrate different input types
	demonstrate_input_types()
}

/*\n   fmt.Printf(\"generation: %d loss: %f. below threshold. breaking\\n\", i, loss)\n   fmt.Printf(\"layer 0, neuron 0\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[0].neurons[0].activation)\n   fmt.Printf(\"layer 0, neuron 1\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[0].neurons[1].activation)\n   fmt.Printf(\"layer 1, neuron 0\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[1].neurons[0].activation)\n   fmt.Printf(\"layer function: %s\\n\", self.networks[0].layers[1].function)\n   fmt.Printf(\"neuron function: %s\\n\", self.networks[0].layers[1].neurons[0].function)\n   fmt.Printf(\"weight1: %f\\n\", self.networks[0].layers[1].neurons[0].synapses[0].weight)\n   fmt.Printf(\"weight2: %f\\n\", self.networks[0].layers[1].neurons[0].synapses[1].weight)\n   fmt.Printf(\"bias: %f\\n\", self.networks[0].layers[1].neurons[0].bias)\n   fmt.Printf(\"layer 1, neuron 1\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[1].neurons[1].activation)\n   fmt.Printf(\"function: %s\\n\", self.networks[0].layers[1].neurons[1].function)\n   fmt.Printf(\"weight1: %f\\n\", self.networks[0].layers[1].neurons[1].synapses[0].weight)\n   fmt.Printf(\"weight2: %f\\n\", self.networks[0].layers[1].neurons[1].synapses[1].weight)\n   fmt.Printf(\"bias: %f\\n\", self.networks[0].layers[1].neurons[1].bias)\n   fmt.Printf(\"layer 2, neuron 0\\n\",)\n   fmt.Printf(\"activation: %f\\n\", self.networks[0].layers[2].neurons[0].activation)\n   fmt.Printf(\"function: %s\\n\", self.networks[0].layers[2].neurons[0].function)\n   fmt.Printf(\"weight1: %f\\n\", self.networks[0].layers[2].neurons[0].synapses[0].weight)\n   fmt.Printf(\"weight2: %f\\n\", self.networks[0].layers[2].neurons[0].synapses[1].weight)\n   fmt.Printf(\"bias: %f\\n\", self.networks[0].layers[2].neurons[0].bias)\n*/
