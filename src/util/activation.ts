//sigmoid function
export function sigmoid(x:number):number {
    return 1 / (1 + Math.exp(-x));
}

//derivative of sigmoid function
export function sigmoidDerivative(x: number): number {
    const k = Math.exp(-x);
    return k / ((1 + k)**2);
}