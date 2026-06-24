// @ts-ignore
import numeric from "numeric";

import dragonObj from "./dragon_points.obj";
import { ObjParser } from "./obj-parser";

export interface RbfSample {
    position: [number, number, number];
    target: number;
}

export enum RbfKernel {
    "linear" = 0,
    "gaussian" = 1,
    "cubic" = 2,
    "quintic" = 3,
    "thin_plate" = 4,
}

export interface RbfFitConfig {
    surfaceSampleCount: number;
    gaussianEpsilon: number;
    sphereRadius: number;
    normalOffset: number;
    regularization: number;
    kernel: RbfKernel;
}

export interface RbfFitResult {
    samples: RbfSample[];
    positions: Float32Array<ArrayBuffer>;
    targets: Float32Array<ArrayBuffer>;
    weights: Float32Array<ArrayBuffer>;
}

export function createBuiltInRbfFit(config: RbfFitConfig): RbfFitResult {
    //const samples: RbfSample[] = createSphereConstraintSamples(config);
    const samples = createObjectConstraintSamples();
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

    const start: number = performance.now();
    const weights: Float32Array<ArrayBuffer> = solveRbfWeights(samples, config);
    const end: number = performance.now();
    console.log(`Linear system solve took: ${end - start} ms`);

    return {
        samples,
        positions,
        targets,
        weights,
    };
}

function createObjectConstraintSamples(): RbfSample[] {
    // --- 1. Downsample e Normalização ---
    const samples: RbfSample[] = [];
    let points = ObjParser.extractPositions(dragonObj);

    // Reduz o número de pontos para algo gerenciável, evitando a explosão de complexidade.
    const maxPoints = 200;
    if (points.length > maxPoints) {
        const downsampledPoints = [];
        const step = Math.floor(points.length / maxPoints);
        for (let i = 0; i < points.length; i += step) {
            downsampledPoints.push(points[i]);
        }
        points = downsampledPoints;
    }
    console.log(`Usando ${points.length} pontos para o RBF fit.`);

    // Normaliza a nuvem de pontos para caber em um raio previsível (ex: 0.7)
    let min = { x: Infinity, y: Infinity, z: Infinity };
    let max = { x: -Infinity, y: -Infinity, z: -Infinity };
    for (const p of points) {
        min.x = Math.min(min.x, p.x);
        min.y = Math.min(min.y, p.y);
        min.z = Math.min(min.z, p.z);
        max.x = Math.max(max.x, p.x);
        max.y = Math.max(max.y, p.y);
        max.z = Math.max(max.z, p.z);
    }
    const center = {
        x: (min.x + max.x) / 2,
        y: (min.y + max.y) / 2,
        z: (min.z + max.z) / 2,
    };
    let maxExtent = 0;
    for (const p of points) {
        const dist = Math.sqrt(
            Math.pow(p.x - center.x, 2) +
                Math.pow(p.y - center.y, 2) +
                Math.pow(p.z - center.z, 2),
        );
        maxExtent = Math.max(maxExtent, dist);
    }
    const desiredRadius = 0.7;
    const scale = desiredRadius / maxExtent;

    const surfacePoints: [number, number, number][] = [];
    for (const p of points) {
        const normalizedPoint: [number, number, number] = [
            (p.x - center.x) * scale,
            (p.y - center.y) * scale,
            (p.z - center.z) * scale,
        ];
        surfacePoints.push(normalizedPoint);
        samples.push({ position: normalizedPoint, target: 0 });
    }

    // --- 2. Adicionar restrições internas e externas ---

    // Adiciona uma âncora interna crucial para definir o "interior" do objeto.
    samples.push({ position: [0, 0, 0], target: -desiredRadius * 0.5 });

    // Mantém as âncoras externas para estabilizar o campo longe do objeto.
    const bounds = desiredRadius * 2.0;
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
            const dist = distance(anchor, sp);
            minDist = Math.min(minDist, dist);
        }
        samples.push({ position: anchor, target: minDist });
    }

    return samples;
}

function createSphereConstraintSamples(config: RbfFitConfig): RbfSample[] {
    const samples: RbfSample[] = [];
    const surfacePoints: [number, number, number][] = [];
    const radius = config.sphereRadius;

    for (let index = 0; index < config.surfaceSampleCount; index += 1) {
        const direction = fibonacciDirection(index, config.surfaceSampleCount);
        const surfacePoint = scale(direction, radius);

        surfacePoints.push(surfacePoint);
        samples.push({ position: surfacePoint, target: 0 });

        if (config.normalOffset > 0) {
            samples.push({
                position: [
                    surfacePoint[0] + direction[0] * config.normalOffset,
                    surfacePoint[1] + direction[1] * config.normalOffset,
                    surfacePoint[2] + direction[2] * config.normalOffset,
                ],
                target: config.normalOffset,
            });

            samples.push({
                position: [
                    surfacePoint[0] - direction[0] * config.normalOffset,
                    surfacePoint[1] - direction[1] * config.normalOffset,
                    surfacePoint[2] - direction[2] * config.normalOffset,
                ],
                target: -config.normalOffset,
            });
        }
    }

    // Ponto âncora interno: Garante que o campo RBF fique negativo no núcleo da esfera
    samples.push({ position: [0, 0, 0], target: -radius });

    // Cria os Anchor Points adaptados dinamicamente ao raio da esfera
    // const bounds = radius * 2.0;
    // const anchorPoints: [number, number, number][] = [
    //     [-bounds, -bounds, -bounds],
    //     [-bounds, -bounds, bounds],
    //     [-bounds, bounds, -bounds],
    //     [-bounds, bounds, bounds],
    //     [bounds, -bounds, -bounds],
    //     [bounds, -bounds, bounds],
    //     [bounds, bounds, -bounds],
    //     [bounds, bounds, bounds],
    // ];

    // for (const anchor of anchorPoints) {
    //     let minDist = Number.MAX_VALUE;
    //     for (const sp of surfacePoints) {
    //         // kernel ou distancia
    //         const dist = distance(anchor, sp);
    //         minDist = Math.min(minDist, dist);
    //     }
    //     samples.push({ position: anchor, target: minDist });
    // }

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
            const diagonal = row === column ? config.regularization : 0;

            matrixRow.push(kernel(radius, config) + diagonal);
        }
        M.push(matrixRow);
    }

    const LU = numeric.LU(M);
    const solvedWeights = numeric.LUsolve(LU, targets);

    return new Float32Array(solvedWeights);
}

function kernel(radius: number, config: RbfFitConfig): number {
    switch (config.kernel) {
        case RbfKernel.linear:
            return radius;
        case RbfKernel.gaussian:
            const scaled = config.gaussianEpsilon * radius;
            return Math.exp(-(scaled * scaled));
        case RbfKernel.cubic:
            return Math.pow(radius, 3);
        case RbfKernel.quintic:
            return Math.pow(radius, 5);
        case RbfKernel.thin_plate:
            if (radius == 0) {
                return 0.0;
            }
            return Math.pow(radius, 2) * Math.log(radius);
        default:
            return radius;
    }
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
