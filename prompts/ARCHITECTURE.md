# Semantic Network Architecture

## Core Components

### WordNeuron
```go
type WordNeuron struct {
    Neuron                    // Embedded base neuron
    word        string        // The word/token this represents
    embedding   []float64     // Word embedding vector (50-dim default)
    frequency   int           // Usage frequency
    context     []string      // Common contextual words
}
```

**Purpose**: Represents individual words/concepts as distinct neural units
**Activation**: Determined by attention mechanism based on context and embedding similarity

### SemanticSynapse
```go
type SemanticSynapse struct {
    Synapse                    // Embedded base synapse
    association_type string    // "co-occurrence", "semantic", "syntactic", "temporal"
    strength         float64   // Learned association strength (0-1)
    confidence       float64   // Confidence in this association
    last_activated   int64     // Timestamp of last activation
}
```

**Purpose**: Represents learned relationships between word concepts
**Evolution**: Strengthens/weakens based on co-occurrence and contextual relevance

### SemanticLayer
```go
type SemanticLayer struct {
    layer_type       string      // "semantic"
    function         string      // activation function ("sigmoid", "relu", "tanh")
    neurons          []WordNeuron // Word neurons in this layer
    attention_matrix [][]float64  // Attention weights between neurons
    context_window   int          // Words to consider in context (default: 3)
}
```

**Purpose**: Manages a collection of word neurons and their attention-based interactions
**Activation Process**:
1. For each word neuron, compute attention scores to context words
2. Use cosine similarity between embeddings as base attention
3. Apply learned attention weights
4. Sum attention contributions and apply activation function

## Information Flow

### Forward Pass (Attention-based Activation)
```
Context Words → Embedding Similarity → Attention Weights → Weighted Sum → Activation Function → Output Activation
```

### Learning Process (Evolutionary)
```
Random Initialization → Fitness Evaluation → Selection → Mutation → Next Generation
```

Where fitness is based on:
- Semantic coherence (predicting appropriate next words)
- Association accuracy (correct relationship strengths)
- Sequence generation quality

## Key Differences from Traditional Neural Networks

| Aspect | Traditional NN | Semantic Network |
|--------|----------------|------------------|
| Units | Feature detectors | Word concepts |
| Connections | Learned weights | Learned associations |
| Activation | Weighted sum + bias | Attention-weighted context |
| Training | Backpropagation | Evolutionary algorithms |
| Architecture | Layered hierarchy | Associative network |
| Purpose | Pattern recognition | Semantic understanding |

## Biological Analogies

- **WordNeurons**: Neural populations representing concepts
- **SemanticSynapses**: Synaptic connections between concept neurons
- **Attention Mechanism**: Top-down modulation of neural activity
- **Embedding Similarity**: Semantic distance between concepts
- **Association Learning**: Hebbian learning ("neurons that fire together wire together")

## Current Limitations

1. **Vocabulary Size**: Currently limited to ~100 words for demo
2. **Training Data**: No real corpus integration yet
3. **Embeddings**: Random initialization instead of pre-trained
4. **Context Scope**: Fixed context window, no long-range dependencies
5. **Association Types**: Basic implementation, not fully utilized

## Extension Points

1. **Multi-layer Networks**: Stack semantic layers for hierarchical understanding
2. **Dynamic Vocabularies**: Add/remove words during training
3. **Attention Heads**: Multiple attention patterns for different relationship types
4. **Memory Systems**: Working memory vs long-term memory distinctions
5. **Meta-learning**: Learn how to learn associations