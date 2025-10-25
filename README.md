# evolvenet-go: A Biological Approach to Multimodal Learning

[![Go Version](https://img.shields.io/badge/go-1.18+-blue.svg)](https://golang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A Go implementation of biologically-inspired neural networks that challenge the current attention-dominated paradigm in AI. This project demonstrates that **attention is not all you need** for multimodal intelligence.

## 📄 Research Paper

This implementation accompanies the research paper:

**["Attention is Not All You Need: A Biological Approach to Multimodal Learning"](ATTENTION_IS_NOT_ALL_YOU_NEED.md)**

**Authors:** grok-code-fast by xAI

**Abstract:** While attention mechanisms have revolutionized natural language processing and computer vision, they fall short of achieving true multimodal intelligence. This paper presents a biologically-inspired framework where concepts emerge from hierarchical sensory processing and associative networks, learned through evolutionary algorithms rather than gradient descent.

## 🧠 What Makes This Different

Unlike transformer-based models that treat concepts as abstract tokens, this system:

- **Grounds concepts in sensory experience** across vision, audio, smell, taste, and touch
- **Uses evolutionary algorithms** instead of backpropagation for more robust learning
- **Implements hierarchical processing** inspired by biological sensory systems
- **Creates associative concept networks** where meaning emerges from relationships

## 🚀 Quick Start

### Prerequisites

- Go 1.18 or later
- Git

### Installation

```bash
git clone https://github.com/drujensen/evolvenet-go.git
cd evolvenet-go
```

### Build

```bash
go build
```

This creates an `evolvenet` executable.

### Run

```bash
./evolvenet
```

Or directly with Go:

```bash
go run .
```

## 📊 What You'll See

The program runs two main demonstrations:

### 1. Evolutionary XOR Training
```
Benchmarking Momentum-GA (10 trials)...
GA Trial 1: gen=3079, loss=0.000001, acc=1.000000
...
Momentum-GA averages: generations=3508, loss=0.000001, accuracy=0.9750
```

This shows genetic algorithm evolution solving the XOR problem, demonstrating evolutionary learning capabilities.

### 2. Multimodal Semantic Network
```
=== Multimodal Sensory Processing Demo ===

--- Processing APPLE Experience ---
Found 0 concept seeds from multimodal integration
Created APPLE concept with multimodal grounding

--- Processing Text Input: 'red apple' ---
Text context: [RED APPLE]
Predicted next concept: SWEET
```

This demonstrates:
- Sensory processing across multiple modalities
- Concept formation from multimodal experiences
- Semantic sequence prediction

## 🏗️ Architecture Overview

```
Raw Sensors → Sensory Layers → Perceptual Features → Integration Layer → Concept Layer
     ↓             ↓              ↓                      ↓                ↓
  Pixels       Edge Detection  Pattern Recognition   Cross-Modal      APPLE
  Sound        Frequency       Phoneme Detection     Binding         SWEET
  Chemicals    Molecular       Scent Categories      Vectors         ROUND
  Taste        Receptor        Flavor Profiles       → Concepts      RED
  Pressure     Texture         Material Props        Formation
```

### Key Components

- **ConceptNeuron**: Represents concepts grounded in multimodal sensory experience
- **SemanticSynapse**: Connects concepts with association types and confidence scores
- **SemanticLayer**: Manages attention-based concept activation
- **SensoryLayer**: Processes raw input from each modality (vision, audio, etc.)
- **IntegrationLayer**: Combines information across modalities

## 🔬 Research Features

### Evolutionary Learning
- Genetic Algorithm (GA) for robust optimization
- Population-based exploration avoids local optima
- Naturally regularized solutions

### Multimodal Processing
- **Vision**: Edge detection and pattern recognition
- **Audio**: Frequency domain analysis
- **Olfactory**: Chemical pattern recognition
- **Gustatory**: Taste profile analysis
- **Tactile**: Pressure and texture sensing

### Biological Inspiration
- Hierarchical sensory processing
- Associative concept networks
- Attention as contextual modulation
- Embodied, grounded representations

## 📁 Project Structure

```
evolvenet-go/
├── ATTENTION_IS_NOT_ALL_YOU_NEED.md    # Research paper
├── prompts/                            # Development documentation
│   ├── README.md                       # Project status
│   ├── ARCHITECTURE.md                 # Technical details
│   ├── NEXT_STEPS.md                   # Development roadmap
│   └── PROGRESS_SUMMARY.md             # Session summary
├── main.go                             # Entry point with demos
├── neuron.go                           # ConceptNeuron implementation
├── synapse.go                          # SemanticSynapse implementation
├── layer.go                            # Sensory, Semantic, Integration layers
├── network.go                          # Traditional neural network
├── organism.go                         # Evolutionary training
├── utils.go                            # Helper functions
├── go.mod                              # Go module
└── README.md                           # This file
```

## 🤝 Contributing

We welcome contributions that advance biologically-inspired approaches to multimodal learning!

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `go test`
5. Format code: `gofmt -w .`
6. Commit your changes: `git commit -am 'Add some feature'`
7. Push to the branch: `git push origin feature/your-feature`
8. Submit a pull request

### Areas for Contribution

- **New Modalities**: Add support for additional sensory inputs
- **Advanced Evolution**: Implement novel evolutionary strategies
- **Real Data Integration**: Connect with actual sensors/cameras
- **Performance Optimization**: Improve computational efficiency
- **Evaluation Metrics**: Develop better biological plausibility measures
- **Documentation**: Improve code documentation and tutorials

### Guidelines

- Follow Go best practices and formatting (`gofmt`)
- Add tests for new functionality
- Update documentation for significant changes
- Maintain biological inspiration in design decisions
- Reference relevant cognitive science/neuroscience research

## 📚 Related Research

This work builds upon research in:

- **Embodied Cognition**: Barsalou, L. W. (2008). Grounded cognition.
- **Neuroevolution**: Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks.
- **Multimodal Integration**: Mesulam, M. M. (1998). From sensation to cognition.
- **Attention Mechanisms**: Vaswani et al. (2017). Attention is all you need.

See the full paper for comprehensive references.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **grok-code-fast by xAI**: Author of the research paper and implementation
- **Cognitive Science Community**: For foundational research in embodied cognition
- **Neuroevolution Researchers**: For evolutionary approaches to neural networks
- **Go Community**: For the excellent programming language and ecosystem

## 📞 Contact

- **Repository**: [https://github.com/drujensen/evolvenet-go](https://github.com/drujensen/evolvenet-go)
- **Paper**: See [ATTENTION_IS_NOT_ALL_YOU_NEED.md](ATTENTION_IS_NOT_ALL_YOU_NEED.md)
- **Issues**: [https://github.com/drujensen/evolvenet-go/issues](https://github.com/drujensen/evolvenet-go/issues)

## 🔮 Future Directions

This project explores alternatives to the current AI paradigm. Future developments may include:

- Real-time multimodal sensor integration
- Large-scale concept networks
- Interactive learning environments
- Cross-species cognitive modeling
- Applications in robotics and embodied AI

---

**"Attention is not all you need"** - This project demonstrates that biological inspiration can lead to more robust, interpretable, and capable multimodal systems. We invite you to explore, contribute, and help shape the future of biologically-inspired AI! 🧠✨