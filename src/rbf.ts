// @ts-ignore
import numeric from "numeric";

export interface RbfSample {
    position: [number, number, number];
    target: number;
}

export interface RbfFitConfig {
    surfaceSampleCount: number;
    gaussianEpsilon: number;
    sphereRadius: number;
}

export interface RbfFitResult {
    samples: RbfSample[];
    positions: Float32Array<ArrayBuffer>;
    targets: Float32Array<ArrayBuffer>;
    weights: Float32Array<ArrayBuffer>;
}

export function createBuiltInRbfFit(config: RbfFitConfig): RbfFitResult {
    const samples: RbfSample[] = createSphereConstraintSamples(
        config.surfaceSampleCount,
        config.sphereRadius,
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
    };
}

function createSphereConstraintSamples(
    surfaceSampleCount: number,
    radius: number,
): RbfSample[] {
    const samples: RbfSample[] = [];
    const surfacePoints: [number, number, number][] = [];

    for (let index = 0; index < surfaceSampleCount; index += 1) {
        const direction = fibonacciDirection(index, surfaceSampleCount);
        const surfacePoint = scale(direction, radius);

        surfacePoints.push(surfacePoint);
        samples.push({ position: surfacePoint, target: 0 });
    }

    // Ponto âncora interno: Garante que o campo RBF fique negativo no núcleo da esfera
    samples.push({ position: [0, 0, 0], target: -radius });

    // Cria os Anchor Points adaptados dinamicamente ao raio da esfera
    const bounds = radius * 2.0;
    const anchorPoints: [number, number, number][] = [
        [-bounds, -bounds, -bounds],
        [-bounds, -bounds, bounds],
        [-bounds, bounds, -bounds],
        [-bounds, bounds, bounds],
        [bounds, -bounds, -bounds],
        [bounds, -bounds, bounds],
        [bounds, bounds, -bounds],
        [bounds, bounds, bounds],
    ];

    for (const anchor of anchorPoints) {
        let minDist = Number.MAX_VALUE;
        for (const sp of surfacePoints) {
            // kernel ou distancia
            const dist = distance(anchor, sp);
            minDist = Math.min(minDist, dist);
        }
        samples.push({ position: anchor, target: minDist });
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
    const M: number[][] = [];
    const targets: number[] = [];

    for (let row = 0; row < count; row += 1) {
        const sampleRow = samples[row];
        targets.push(sampleRow.target);
        const matrixRow: number[] = [];

        for (let column = 0; column < count; column += 1) {
            const sampleColumn = samples[column];
            const radius = distance(sampleRow.position, sampleColumn.position);
            //const diagonal = row === column ? config.regularization : 0;
            const diagonal = 0;

            matrixRow.push(
                gaussianKernel(radius, config.gaussianEpsilon) + diagonal,
            );
        }
        M.push(matrixRow);
    }

    const LU = numeric.LU(M);
    const solvedWeights = numeric.LUsolve(LU, targets);

    return new Float32Array(solvedWeights);
}

function gaussianKernel(radius: number, epsilon: number): number {
    const scaled = epsilon * radius;
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
