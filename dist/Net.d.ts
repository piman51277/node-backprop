declare type NetConstructor = {
    weights: number[][];
    biases: number[][];
    dimensions: number[];
};
declare type Gradient = [number[][], number[][]];
export declare type TrainingData = [number[], number[]][];
declare type TrainingConfig = {
    gamma?: number;
    gamma_b?: number;
    momentum?: number;
    batchSize?: number;
};
export declare class Net {
    weights: number[][];
    biases: number[][];
    dimensions: number[];
    constructor({ weights, biases, dimensions }: NetConstructor);
    /**
     * Creates a new net with the given dimensions.
     * @param inputSize The size of the input layer.
     * @param outputSize The size of the output layer.
     * @param hiddenSize The size of the hidden layers.
     * @param hiddenLayers The number of hidden layers.
     * @returns
     */
    static create(inputSize: number, outputSize: number, hiddenSize: number, hiddenLayers: number): Net;
    static mergeGradients(gradients: Gradient[]): Gradient;
    /**
     * Evaluates the net with the given input.
     * @param input Input to the net.
     * @returns Populated nodes of the net.
     */
    eval(input: number[]): number[][];
    /**
     * Computes the gradient of the net with the given input and expected output.
     * @param input input to the net.
     * @param expected expected output of the net.
     * @returns gradient of the net.
     */
    gradient(input: number[], expected: number[]): Gradient;
    /**
     * Apply the gradient to the net.
     * @param gradient gradient of the net.
     */
    apply([sigmas, weightGradients]: [number[][], number[][]], gamma?: number, gamma_b?: number): void;
    /**
     * Gets the error of the net with the given input and expected output. USE ONLY FOR DEBUG
     * @param input input to the net.
     * @param expected expected output of the net.
     * @returns error of the net.
     */
    error(input: number[], expected: number[]): number;
    /**
     * Average of error of the net with the given input and expected output. USE ONLY FOR DEBUG
     * @param data Array containing test data.
     */
    errorDataset(data: TrainingData): number;
    /**
     * Prints the net.
     */
    debug(): void;
    /**
     * Trains the net with the given input and expected output.
     * @param data Training data.
     * @param epochs Number of epochs to train the net.
     * @param config Configuration of the training.
     */
    train(data: TrainingData, epochs: number, config?: TrainingConfig): void;
}
export {};
