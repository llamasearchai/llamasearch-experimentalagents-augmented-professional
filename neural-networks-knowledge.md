# Deep Dive into Neural Networks

## Fundamentals of Neural Networks

Neural networks are computational systems inspired by the biological neural networks that constitute animal brains. At their core, they are made up of neurons, or nodes, which are interconnected and organized in layers. These systems "learn" to perform tasks by considering examples, generally without being programmed with task-specific rules.

The basic structure includes:

- **Input Layer**: Receives initial data (like pixel values in an image)
- **Hidden Layers**: Intermediate layers that transform the data
- **Output Layer**: Produces the final result (like classification probabilities)

Each neuron applies a transformation to its inputs, typically consisting of:
1. A weighted sum of inputs
2. A bias term addition
3. An activation function that introduces non-linearity

## Activation Functions

Activation functions determine the output of a neural network by defining how the weighted sum of inputs is transformed into an output. Common activation functions include:

- **Sigmoid**: Maps values to range (0,1), historically popular but suffers from vanishing gradient issues
- **Tanh**: Maps values to range (-1,1), similar to sigmoid but centered at zero
- **ReLU (Rectified Linear Unit)**: Returns x if x > 0, else 0; most widely used due to computational efficiency
- **Leaky ReLU**: Returns x if x > 0, else αx (where α is a small constant); addresses "dying ReLU" problem
- **Softmax**: Used in output layer for multi-class classification; converts logits to probabilities

## Training Neural Networks

The training process for neural networks involves:

1. **Forward Propagation**: Input data passes through the network to generate an output
2. **Loss Calculation**: The difference between predicted output and actual target is measured
3. **Backward Propagation**: Gradients of the loss with respect to weights are calculated
4. **Weight Updates**: Weights are adjusted to minimize loss using an optimizer

Key concepts in training include:

- **Learning Rate**: Controls how much weights are updated during training
- **Batch Size**: Number of examples processed before weight update
- **Epoch**: One complete pass through the entire training dataset
- **Regularization**: Techniques to prevent overfitting (e.g., L1/L2 regularization, dropout)

## Types of Neural Networks

Various neural network architectures serve different purposes:

### Feedforward Neural Networks (FNN)
The simplest type where information flows in one direction from input to output. Good for tabular data and simple pattern recognition.

### Convolutional Neural Networks (CNN)
Specialized for processing grid-like data such as images. Key components include:
- Convolutional layers that apply filters to detect features
- Pooling layers that reduce dimensionality
- Fully connected layers that perform classification

### Recurrent Neural Networks (RNN)
Designed for sequential data by maintaining a memory of previous inputs. Variants include:
- Long Short-Term Memory (LSTM): Addresses vanishing gradient problem
- Gated Recurrent Unit (GRU): Simplified version of LSTM
- Bidirectional RNNs: Process sequences in both forward and backward directions

### Transformer Networks
Introduced the attention mechanism