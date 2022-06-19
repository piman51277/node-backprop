export function generateRandom(min: number, max: number, size: number): number[] {
    const result: number[] = [];
    for (let i = 0; i < size; i++) {
        result.push(Math.random() * (max - min) + min);
    }
    return result;
}