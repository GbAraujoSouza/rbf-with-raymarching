export interface RbfSample {
    position: [number, number, number];
    target: number;
}

export interface RbfFitConfig {
    radius: number;
    offsetDistance: number;
    surfaceSampleCount: number;
    kernelSigma: number;
    regularization: number;
}

export interface RbfFitResult {
    samples: RbfSample[];
    positions: Float32Array<ArrayBuffer>;
    targets: Float32Array<ArrayBuffer>;
    weights: Float32Array<ArrayBuffer>;
    kernelSigma: number;
    correctionPower: number;
    correctionLinear: number;
    pointRadius: number;
}

const DEFAULT_CONFIG: RbfFitConfig = {
    radius: 1,
    offsetDistance: 0.18,
    surfaceSampleCount: 18,
    kernelSigma: 1.35,
    regularization: 1e-5,
};

const CORRECTION_POWER = 0.85;
const CORRECTION_LINEAR = 0.9;
const DEBUG_POINT_RADIUS = 0.06;

export function createBuiltInRbfFit(
    overrides: Partial<RbfFitConfig> = {},
): RbfFitResult {
    const config = { ...DEFAULT_CONFIG, ...overrides };
    const samples = createSphereConstraintSamples(
        config.surfaceSampleCount,
        config.radius,
        config.offsetDistance,
    );
    const targets = new Float32Array(samples.length);
    const positions = new Float32Array(samples.length * 4);

    for (let index = 0; index < samples.length; index += 1) {
        const sample = samples[index];
        targets[index] = sample.target;
        positions[index * 4 + 0] = sample.position[0];
        positions[index * 4 + 1] = sample.position[1];
        positions[index * 4 + 2] = sample.position[2];
        positions[index * 4 + 3] = 0;
    }

    const weights = solveRbfWeights(samples, config);

    return {
        samples,
        positions,
        targets,
        weights,
        kernelSigma: config.kernelSigma,
        correctionPower: CORRECTION_POWER,
        correctionLinear: CORRECTION_LINEAR,
        pointRadius: DEBUG_POINT_RADIUS,
    };
}

function createSphereConstraintSamples(
    surfaceSampleCount: number,
    radius: number,
    offsetDistance: number,
): RbfSample[] {
    const samples: RbfSample[] = [];

    for (let index = 0; index < surfaceSampleCount; index += 1) {
        const direction = fibonacciDirection(index, surfaceSampleCount);
        const surfacePoint = scale(direction, radius);
        const outwardPoint = scale(direction, radius + offsetDistance);
        const inwardPoint = scale(direction, radius - offsetDistance);

        samples.push({ position: surfacePoint, target: 0 });
        samples.push({ position: outwardPoint, target: offsetDistance });
        samples.push({ position: inwardPoint, target: -offsetDistance });
    }

    return samples;
}

function fibonacciDirection(
    index: number,
    total: number,
): [number, number, number] {
    const offset = 2 / total;
    const y = index * offset - 1 + offset / 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const theta = index * goldenAngle;

    return [Math.cos(theta) * radius, y, Math.sin(theta) * radius];
}

function solveRbfWeights(
    samples: RbfSample[],
    config: RbfFitConfig,
): Float32Array<ArrayBuffer> {
    const count = samples.length;
    const system = new Float64Array(count * count);
    const targets = new Float64Array(count);

    for (let row = 0; row < count; row += 1) {
        const sampleRow = samples[row];
        targets[row] = sampleRow.target;

        for (let column = 0; column < count; column += 1) {
            const sampleColumn = samples[column];
            const radius = distance(sampleRow.position, sampleColumn.position);
            const diagonal = row === column ? config.regularization : 0;

            system[row * count + column] =
                gaussianKernel(radius, config.kernelSigma) + diagonal;
        }
    }

    const solvedWeights = solveLinearSystem(system, targets, count);
    return new Float32Array(solvedWeights);
}

function solveLinearSystem(
    matrix: Float64Array<ArrayBuffer>,
    vector: Float64Array<ArrayBuffer>,
    size: number,
): Float64Array<ArrayBuffer> {
    const augmented = new Float64Array(size * (size + 1));

    for (let row = 0; row < size; row += 1) {
        for (let column = 0; column < size; column += 1) {
            augmented[row * (size + 1) + column] = matrix[row * size + column];
        }
        augmented[row * (size + 1) + size] = vector[row];
    }

    for (let pivot = 0; pivot < size; pivot += 1) {
        let maxRow = pivot;
        let maxValue = Math.abs(augmented[pivot * (size + 1) + pivot]);

        for (let row = pivot + 1; row < size; row += 1) {
            const value = Math.abs(augmented[row * (size + 1) + pivot]);
            if (value > maxValue) {
                maxValue = value;
                maxRow = row;
            }
        }

        if (maxValue < 1e-10) {
            throw new Error("RBF system is singular or ill-conditioned.");
        }

        if (maxRow !== pivot) {
            swapRows(augmented, size + 1, pivot, maxRow);
        }

        const pivotValue = augmented[pivot * (size + 1) + pivot];

        for (let column = pivot; column <= size; column += 1) {
            augmented[pivot * (size + 1) + column] /= pivotValue;
        }

        for (let row = 0; row < size; row += 1) {
            if (row === pivot) {
                continue;
            }

            const factor = augmented[row * (size + 1) + pivot];
            if (factor === 0) {
                continue;
            }

            for (let column = pivot; column <= size; column += 1) {
                augmented[row * (size + 1) + column] -=
                    factor * augmented[pivot * (size + 1) + column];
            }
        }
    }

    const solution = new Float64Array(size);
    for (let row = 0; row < size; row += 1) {
        solution[row] = augmented[row * (size + 1) + size];
    }

    return solution;
}

function swapRows(
    matrix: Float64Array<ArrayBuffer>,
    stride: number,
    rowA: number,
    rowB: number,
) {
    for (let column = 0; column < stride; column += 1) {
        const firstIndex = rowA * stride + column;
        const secondIndex = rowB * stride + column;
        const temporary = matrix[firstIndex];
        matrix[firstIndex] = matrix[secondIndex];
        matrix[secondIndex] = temporary;
    }
}

function gaussianKernel(radius: number, sigma: number): number {
    const scaled = sigma * radius;
    return Math.exp(-(scaled * scaled));
}

function distance(
    first: [number, number, number],
    second: [number, number, number],
): number {
    const dx = first[0] - second[0];
    const dy = first[1] - second[1];
    const dz = first[2] - second[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function scale(
    vector: [number, number, number],
    factor: number,
): [number, number, number] {
    return [vector[0] * factor, vector[1] * factor, vector[2] * factor];
}
