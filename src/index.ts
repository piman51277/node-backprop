import { TrainingData, Net } from "./Net";
//XOR
const trainingSet: TrainingData = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]],
];
const net = Net.create(2, 1, 3, 1);

console.log(net.errorDataset(trainingSet));
const trainingConfig = {
  gamma: 0.1,
  gamma_b: 0.1,
  momentum: 0.8,
};
net.train(trainingSet, 1000000, trainingConfig);
console.log(net.errorDataset(trainingSet));