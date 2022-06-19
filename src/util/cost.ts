//MSE
export function MSE(actual: number[], expected: number[]) {
    return actual.reduce((acc, curr, i) => {
        return acc + Math.pow(curr - expected[i], 2);
    }, 0) * (1 / actual.length);
}
    