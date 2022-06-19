//XOR
const trainingSet = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]],
];
import { Net } from "./Net";
const net = new Net({
  weights: [
    [0.15, 0.2, 0.25, 0.3],
    [0.4, 0.45, 0.5, 0.55],
  ],
  biases: [
    [0.35, 0.35],
    [0.6, 0.6],
  ],
  dimensions: [2, 2, 2],
});

//console.log(net.eval([0.05, 0.10]));
console.log(net.gradient([0.05, 0.10], [0.01, 0.99]));
 