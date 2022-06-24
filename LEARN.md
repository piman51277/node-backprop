# Project Info

While the README focuses on documentation, this file contains information about the project.  
Information about the algorithm itself is ommitted, as it is better explained by others.

# Project Description

This project is an optimised implementation of a feed-forward neural network, trained using the backpropagation algorithm, with adjustable momentum and learning rate (gamma). This version was written in Typescript.  
Useful Links:
 - Backpropagation: [Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)
 - Feed-forward neural network: [Wikipedia](https://en.wikipedia.org/wiki/Feedforward_neural_network)

# Implementation

The entirety of this project is implementated as a single class: `Net`. Aside from utility functions, this class has three main methods:
- `eval` which takes inputs and evaluates the entire network
- `gradient` which takes inputs and expected outputs and calculates the gradient of the network
- `apply` which takes a gradient and updates the weights and biases of the network

Momentum is implemented in the `train` method, which repeatedly calles `gradient` and `apply` for a specified number of epochs.

# Optimizations

1. This project was written in Typescript, and configured to transpile to a relatively modern version of Javascript (es2017). This means that slow polyfills will not be added to the code during transpilation. In addition, type safety was enforced, allowing optimization by TurboFan. [config file](/tsconfig.json#L3)
2. Instead of using 2D arrays to store weight values, this project uses 1D arrays. There are multiple reasons this is faster, but the most important one is that it does not require two access operations to get the wanted value.
3. In addition to #2, weight values are stored in the order they are used, meaning we do not have to calculate the index of a weight multiple math operations (`row * width + col`), but rather a simple counter.
4. Function calls were minimised. In Javascript, calling a function is a very expensive operation because a new function context needs to be created for each call. This means functional programming was avoided in performance-critical functions.
   - If you want to read more about the internals of Javascript, read this [FreeCodeCamp article](https://www.freecodecamp.org/news/javascript-under-the-hood-v8/) 
