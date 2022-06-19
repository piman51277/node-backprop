import { sigmoid } from "./util/activation";
import { MSE } from "./util/cost";
import { generateRandom } from "./util/generateRandom";

type NetConstructor = {
  weights: number[][];
  biases: number[][];
  dimensions: number[];
};
export class Net {
  weights: number[][]; //contains all weights
  biases: number[][]; //contains all biases
  dimensions: number[]; //contains the size of each layer

  //constructor
  constructor({ weights, biases, dimensions }: NetConstructor) {
    this.weights = weights;
    this.biases = biases;
    this.dimensions = dimensions;
  }

  /**
   * Creates a new net with the given dimensions.
   * @param inputSize The size of the input layer.
   * @param outputSize The size of the output layer.
   * @param hiddenSize The size of the hidden layers.
   * @param hiddenLayers The number of hidden layers.
   * @returns
   */
  static create(
    inputSize: number,
    outputSize: number,
    hiddenSize: number,
    hiddenLayers: number
  ) {
    //weights
    const inputWeights = generateRandom(-1, 1, inputSize * hiddenSize);
    const outputWeights = generateRandom(-1, 1, hiddenSize * outputSize);
    const hiddenWeights = new Array(hiddenLayers - 1)
      .fill(0)
      .map(() => generateRandom(-1, 1, hiddenSize * hiddenSize));

    const weights = [inputWeights, ...hiddenWeights, outputWeights];

    //biases
    const outputBiases = generateRandom(-1, 1, outputSize);
    const hiddenBiases = new Array(hiddenLayers)
      .fill(0)
      .map(() => generateRandom(-1, 1, hiddenSize));

    const biases = [...hiddenBiases, outputBiases];

    //dimensions
    const dimensions = [
      inputSize,
      ...new Array(hiddenLayers).fill(hiddenSize),
      outputSize,
    ];

    return new Net({
      weights,
      biases,
      dimensions,
    });
  }

  /**
   * Evaluates the net with the given input.
   * @param input Input to the net.
   * @returns Populated nodes of the net.
   */
  eval(input: number[]): number[][] {
    //check if the input is the correct size
    if (input.length !== this.dimensions[0]) {
      throw new Error(
        `Input size ${input.length} does not match net input size ${this.dimensions[0]}`
      );
    }

    //start evaluating the net
    const nodes = [input];

    for (let layer = 1; layer < this.dimensions.length; layer++) {
      const lastLayerSize = this.dimensions[layer - 1];
      const layerSize = this.dimensions[layer];
      const lastLayerNodes = nodes[layer - 1];

      //compute the nodes of the current layer
      const layerNodes = new Array(layerSize).fill(0);

      let weightPosPrecompute = 0; //equivalent to [node * layerSize + lastLayerSize] or [layerSize][lastLayerSize]
      for (let node = 0; node < layerSize; node++) {
        let sum = 0;

        for (let i = 0; i < lastLayerSize; i++) {
          sum +=
            lastLayerNodes[i] * this.weights[layer - 1][weightPosPrecompute];
          weightPosPrecompute++;
        }
        layerNodes[node] = sigmoid(sum + this.biases[layer - 1][node]);
      }

      nodes.push(layerNodes);
    }

    //return the output
    return nodes;
  }

  /**
   * Computes the gradient of the net with the given input and expected output.
   * @param input input to the net.
   * @param expected expected output of the net.
   * @returns gradient of the net.
   */
  gradient(input: number[], expected: number[]): [number[][], number[][]] {
    const outputs = this.eval(input).reverse();
    const weights = [...this.weights].reverse();
    const dimensions = [...this.dimensions].reverse();
    const sigmas: number[][] = [];
    const weightGradients: number[][] = [];

    //compute the sigma values for the last layer
    const lastSigma = [];
    for (let i = 0; i < dimensions[0]; i++) {
      lastSigma.push(
        outputs[0][i] * (1 - outputs[0][i]) * (outputs[0][i] - expected[i])
      );
    }
    sigmas.push(lastSigma);

    //compute the sigma values for the other layers
    for (let layer = 1; layer < dimensions.length; layer++) {
      const lastLayerSigma = sigmas[layer - 1];
      const thisLayerSigma = [];
      const weightGradient = [];
      for (let node = 0; node < dimensions[layer]; node++) {
        let sum = 0;
        for (let nextNode = 0; nextNode < dimensions[layer - 1]; nextNode++) {
          sum +=
            weights[layer - 1][node + dimensions[layer - 1] * nextNode] *
            lastLayerSigma[nextNode];

          weightGradient.push(lastLayerSigma[nextNode] * outputs[layer][node]);
        }
        thisLayerSigma.push(
          sum * (1 - outputs[layer][node]) * outputs[layer][node]
        );
      }
      sigmas.push(thisLayerSigma);
      weightGradients.push(weightGradient);
    }

    return [sigmas.reverse(), weightGradients.reverse()];
  }

  /**
   * Apply the gradient to the net.
   * @param gradient gradient of the net.
   */
  apply(
    [sigmas, weightGradients]: [number[][], number[][]],
    gamma = 0.01,
    gamma_b = 0.01
  ) {
    for (let layer = 1; layer < sigmas.length; layer++) {
      for (let node = 0; node < sigmas[layer].length; node++) {
        this.biases[layer - 1][node] += sigmas[layer][node] * gamma_b;
      }
      this.weights[layer - 1] = this.weights[layer - 1].map((weight, i) => {
        return weight + weightGradients[layer - 1][i] * gamma;
      });
    }
  }

  /**
   * Gets the error of the net with the given input and expected output. USE ONLY FOR DEBUG
   * @param input input to the net.
   * @param expected expected output of the net.
   * @returns error of the net.
   */
  error(input: number[], expected: number[]): number {
    const actual = this.eval(input)[this.dimensions.length - 1];
    return MSE(actual, expected);
  }
  /**
   * Prints the net.
   */
  debug(): void {
    console.log("Weights: ");
    console.log(this.weights.map((n) => n.toString()).join("\n"));
    console.log("Biases: ");
    console.log(this.biases.map((n) => n.toString()).join("\n"));
  }
}
