package main

import (
  "fmt"
  "math"
  "math/rand"
  "sort"
  "time"
  "strings"
)

// utility functions
// =============================================================================
func randFloat(min, max float64) float64 {
  return min + rand.Float64()*(max-min)
}

func roundFloat(f float64, n int) float64 {
  return math.Round(f*math.Pow10(n)) / math.Pow10(n)
}

// =============================================================================
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

// =============================================================================
type Neuron struct {
  function   string
  activation float64
  bias       float64
  synapses   []Synapse
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

// =============================================================================
type Layer struct {
  layer_type string
  function   string
  neurons    []Neuron
}

func (self *Layer) clone(other *Layer) {
  self.layer_type = strings.Clone(other.layer_type)
  self.function = strings.Clone(other.function)

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

// =============================================================================
type Network struct {
  loss   float64
  layers []Layer
}

func (self *Network) add_layer(layer_type string, function string, size int) {
  layer := Layer{strings.Clone(layer_type), strings.Clone(function), make([]Neuron, size)}
  self.layers = append(self.layers, layer)
}

func (self *Network) fully_connect() {
  for i := 0; i < len(self.layers); i++ {
    if self.layers[i].layer_type == "hidden" || self.layers[i].layer_type == "output" {
      for j := 0; j < len(self.layers[i].neurons); j++ {
        self.layers[i].neurons[j].function = strings.Clone(self.layers[i].function)
        for k := 0; k < len(self.layers[i-1].neurons); k++ {
          self.layers[i].neurons[j].synapses = append(self.layers[i].neurons[j].synapses, Synapse{0.5, k})
        }
      }
    }
  }
}

func (self *Network) clone(other *Network) {
  self.loss = other.loss

  if len(self.layers) != len(other.layers) {
    self.layers = make([]Layer, len(other.layers))
  }

  for i := 0; i < len(self.layers); i++ {
    self.layers[i].clone(&other.layers[i])
  }
}

func (self *Network) randomize() {
  self.loss = 1.0
  for i := 0; i < len(self.layers); i++ {
    self.layers[i].randomize()
  }
}

func (self *Network) mutate() {
  for i := 0; i < len(self.layers); i++ {
    self.layers[i].mutate(0.1)
  }
}

func (self *Network) punctuate(pos int) {
  for i := 0; i < len(self.layers); i++ {
    self.layers[i].punctuate(pos)
  }
}

func (self *Network) run(data []float64) []float64 {
  for i := 0; i < len(self.layers); i++ {
    if i == 0 {
      self.layers[i].set_activations(data)
    } else {
      self.layers[i].activate(&self.layers[i-1])
    }
  }

  return self.layers[len(self.layers)-1].get_activations()
}

func (self *Network) evaluate(data [][][]float64) {
  sum := 0.0
  for i := 0; i < len(data); i++ {
    actual := self.run(data[i][0])
    expected := data[i][1]
    for j := 0; j < len(expected); j++ {
      sum += math.Pow(expected[j] - actual[j], 2)
    }
  }
  self.loss = float64(sum / (2.0 * float64(len(data))))
}

func (self *Network) get_loss() float64 {
  return self.loss
}

// =============================================================================
type Organism struct {
  networks []Network
}

func (self *Organism) init(network *Network, size int) {
  self.networks = make([]Network, size)
  for i := 0; i < size; i++ {
    self.networks[i].clone(network)
    self.networks[i].randomize()
  }
}

func (self *Organism) evolve(data [][][]float64, generations int, threshold float64, log_every int) *Network {
  half := len(self.networks) / 2
  quarter := half / 2

  // for every generation
  for i := 0; i < generations; i++ {

    // evaluate the networks
    for j := 0; j < len(self.networks); j++ {
      self.networks[j].evaluate(data)
    }

    // sort the networks by loss
    sort.Slice(self.networks, func(i, j int) bool { return self.networks[i].get_loss() < self.networks[j].get_loss() })

    loss := self.networks[0].get_loss()
    if threshold > 0 && loss < threshold {
      break
    }

    if i%log_every == 0 {
      fmt.Printf("generation: %d loss: %f\n", i, loss)
    }

    // clone the top quarter to the bottom quarter of the networks
    for j := 0; j < quarter; j++ {
      self.networks[half+quarter+j].clone(&self.networks[j])
    }

    // punctuate several of the networks
    for j := 1; j < quarter; j++ {
      self.networks[j].punctuate(j)
    }

    // mutate all but the best and punctuated networks
    for j := quarter; j < len(self.networks); j++ {
      self.networks[j].mutate()
    }
  }
  // sort the networks by fitness
  sort.Slice(self.networks, func(i, j int) bool { return self.networks[i].get_loss() < self.networks[j].get_loss() })
  return &self.networks[0]
}

// =============================================================================
func init() {
  rand.Seed(time.Now().UnixNano())
}

func main() {
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
  organism.init(network, 16)

  network = organism.evolve(data, 10000000, 0.000001, 1000)
  tn, tp, fn, fp, ct := 0, 0, 0, 0, 0

  for i := 0; i < len(data); i++ {
    actual := network.run(data[i][0])
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

  fmt.Printf("Test size: %d\n", len(data))
  fmt.Printf("----------------------\n")
  fmt.Printf("TN: %d | FP: %d\n", tn, fp)
  fmt.Printf("----------------------\n")
  fmt.Printf("FN: %d | TP: %d\n", fn, tp)
  fmt.Printf("----------------------\n")
  fmt.Printf("Accuracy: %f\n", float64(tn+tp)/float64(ct))
}

/*
      fmt.Printf("generation: %d loss: %f. below threshold. breaking\n", i, loss)
      fmt.Printf("layer 0, neuron 0\n",)
      fmt.Printf("activation: %f\n", self.networks[0].layers[0].neurons[0].activation)
      fmt.Printf("layer 0, neuron 1\n",)
      fmt.Printf("activation: %f\n", self.networks[0].layers[0].neurons[1].activation)
      fmt.Printf("layer 1, neuron 0\n",)
      fmt.Printf("activation: %f\n", self.networks[0].layers[1].neurons[0].activation)
      fmt.Printf("layer function: %s\n", self.networks[0].layers[1].function)
      fmt.Printf("neuron function: %s\n", self.networks[0].layers[1].neurons[0].function)
      fmt.Printf("weight1: %f\n", self.networks[0].layers[1].neurons[0].synapses[0].weight)
      fmt.Printf("weight2: %f\n", self.networks[0].layers[1].neurons[0].synapses[1].weight)
      fmt.Printf("bias: %f\n", self.networks[0].layers[1].neurons[0].bias)
      fmt.Printf("layer 1, neuron 1\n",)
      fmt.Printf("activation: %f\n", self.networks[0].layers[1].neurons[1].activation)
      fmt.Printf("function: %s\n", self.networks[0].layers[1].neurons[1].function)
      fmt.Printf("weight1: %f\n", self.networks[0].layers[1].neurons[1].synapses[0].weight)
      fmt.Printf("weight2: %f\n", self.networks[0].layers[1].neurons[1].synapses[1].weight)
      fmt.Printf("bias: %f\n", self.networks[0].layers[1].neurons[1].bias)
      fmt.Printf("layer 2, neuron 0\n",)
      fmt.Printf("activation: %f\n", self.networks[0].layers[2].neurons[0].activation)
      fmt.Printf("function: %s\n", self.networks[0].layers[2].neurons[0].function)
      fmt.Printf("weight1: %f\n", self.networks[0].layers[2].neurons[0].synapses[0].weight)
      fmt.Printf("weight2: %f\n", self.networks[0].layers[2].neurons[0].synapses[1].weight)
      fmt.Printf("bias: %f\n", self.networks[0].layers[2].neurons[0].bias)
*/
