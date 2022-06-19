# backprop

An optimized implementation of the backpropagation algorithm in Typescript

# Example Usage

```typescript
import { TrainingData, Net } from "./Net";

//example dataset
const XOR: TrainingData = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]],
];

//create a net with random weights/biases
const net = Net.create(2, 1, 3, 1);

//train the net
const trainConfig = {
  gamma: 0.1, //learning rate of weights
  gamma_b: 0.1, //learning rate of biases
  momentum: 0.1, //momentum of weights/biases
  batchSize: 1, //batch size
};

console.log(net.errorDataset(XOR)); //print the average error of the net
net.train(XOR, 100000, trainConfig); //train the net for 100000 epochs
console.log(net.errorDataset(XOR)); //print the new average error of the net
```

# Documentation

## Table of Contents

## [Net](#net-1)

- [Net.create](#netcreateinputs-outputs-hidden-hiddenlayers)
- [Net.mergeGradients](#netmergegradientsgradients)

## [Net.prototype](#netprototype-1)

- [Net.prototype.eval](#netprototypeevalinput)
- [Net.prototype.gradient](#netprototypegradientinput-expected)
- [Net.prototype.apply](#netprototypeapplygradient--gamma01-gammab01)
- [Net.prototype.train](#netprototypetraindataset-epochs-options)
- [Net.prototype.error](#netprototypeerrordatasetdataset)
- [Net.prototype.errorDataset](#netprototypeerrordatasetdataset)
- [Net.prototype.debug](#netprototypedebug)

## [Misc](#misc-1)

- [`NetOptions`](#netoptions-1)
- [`Gradient`](#gradient)
- [`TrainingData`](#trainingdata)
- [`TrainingOptions`](#trainingoptions)

## `Net`

### `Net(options)`

[source](./src/Net.ts#L32) | [TOC](#table-of-contents)  
Constructor for the Net class.

#### Arguments

- `options` {[NetOptions](#netoptions-1)} - The weights and biases of the net.

### `Net.create(inputs, outputs, hidden, hiddenLayers)`

[source](./src/Net.ts#L46) | [TOC](#table-of-contents)  
Creates a new net with the given dimensions.

#### Arguments

- `inputs` {`number`}: The size of the input layer.
- `outputs` {`number`}: The size of the output layer.
- `hidden` {`number`}: The size of the hidden layer.
- `hiddenLayers` {`number`}: The number of hidden layers.

#### Returns

{`Net`}: An instance of the `Net` class, with random weights and biases.

### `Net.mergeGradients(gradients)`

[source](./src/Net.ts#L83) | [TOC](#table-of-contents)  
Average several gradients into one.

#### Arguments

- `gradients` {[Gradient](#gradient)`[]`}: The gradients to average.

#### Returns

{[Gradient](#gradient)}: The averaged gradient.

## `Net.prototype`

### `Net.prototype.eval(input)`

[source](./src/Net.ts#L113) | [TOC](#table-of-contents)  
Run a forward pass of the net using the given input.

#### Arguments

- `input` {`number[]`}: The input to the net.

#### Returns

{`number[][]`}: The state of the net after the input has been passed.

### `Net.prototype.gradient(input, expected)`

[source](./src/Net.ts#L158) | [TOC](#table-of-contents)  
Calculate the required gradient for a given test case

#### Arguments

- `input` {`number[]`}: The input to the net.
- `expected` {`number[]`}: The expected output of the net.

#### Returns

{[Gradient](#gradient)}: The gradient of the net.

### `Net.prototype.apply(gradient [, gamma=0.1, gamma_b=0.1])`

[source](./src/Net.ts#L204) | [TOC](#table-of-contents)  
Apply a gradient to the net.

#### Arguments

- `gradient` {[Gradient](#gradient)`[]`}: The gradient to apply.
- `gamma` {`number`}: [Optional] The learning rate of the weights.
- `gamma_b` {`number`}: [Optional] The learning rate of the biases.

### `Net.prototype.train(dataset, epochs, options)`

[source](./src/Net.ts#L264) | [TOC](#table-of-contents)  
Train the net using the given dataset.

#### Arguments

- `dataset` {[TrainingData](#trainingdata)`[]`}: The dataset to train the net on.
- `epochs` {`number`}: The number of epochs to train the net for.
- `options` {[TrainingOptions](#trainingoptions)}: The options for training.

### `Net.prototype.error(input, expected)`

[source](./src/Net.ts#L226) | [TOC](#table-of-contents)  
Calculate the error of the net for a given test case using MSE

#### Arguments

- `input` {`number[]`}: The input to the net.
- `expected` {`number[]`}: The expected output of the net.

#### Returns

{`number`}: The error of the net.

### `Net.prototype.errorDataset(dataset)`

[source](./src/Net.ts#L240) | [TOC](#table-of-contents)  
Calculate the average error of the net for a given dataset.

#### Arguments

- `dataset` {[TrainingData](#trainingdata)`[]`}: The dataset to use.

#### Returns

{`number`}: The average error of the net.

### `Net.prototype.debug()`

[source](./src/Net.ts#L251) | [TOC](#table-of-contents)  
Prints the state of the net to the console.

#### Format

```
Weights:
<Layer0 Weight0>, <Layer0 Weight1>, ..., <Layer0 WeightN>
<Layer1 Weight0>, <Layer1 Weight1>, ..., <Layer1 WeightN>
...
<LayerM Weight0>, <LayerM Weight1>, ..., <LayerM WeightN>
Biases:
<Layer0 Bias0>, <Layer0 Bias1>, ..., <Layer0 BiasN>
<Layer1 Bias0>, <Layer1 Bias1>, ..., <Layer1 BiasN>
...
<LayerM Bias0>, <LayerM Bias1>, ..., <LayerM BiasN>
```

## `Misc`

### `NetOptions`

```typescript
{
  weights: number[][];
  biases: number[][];
  dimensions: number[];
};
```

An object describing the weights and biases of a net.

### `Gradient`

```typescript
[
    number[][], //bias gradient
    number[][] //weight gradient
]
```

An object describing the gradient of a net.

### `TrainingData`

```typescript
[
    [number[], number[]], //input, expected output
    [number[], number[]],
    [number[], number[]],
    ...
]
```

An array of training data.

### `TrainingOptions`

```typescript
{
  gamma?: number; //learning rate of the weights
  gamma_b?: number; //learning rate of the biases
  momentum?: number; //momentum of the weights/biases
  batchSize?: number; //the size of the batch
}
```

# Notes

## Weight Configuration

Weights are internally stored as a 1D array, grouped by origin node. This is so the weights can be read sequentially, increasing the performance of the network.

### Example

![A diagram of a simple net](/readme/examplenet.png)

For this net, the weights would be stored as such:

```typescript
[
  [0.4, 0.8, 0.52, 0.3], //Layer 0
  [0.1, 0.63, 0.48, 0.2], //Layer 1
];
```
