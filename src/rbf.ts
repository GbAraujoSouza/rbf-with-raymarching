// @ts-ignore
import numeric from "numeric";

import bunnyPly from "./models/bunny/bunny/reconstruction/bun_zipper.ply";
import dragonPly from "./models/dragon_recon/dragon_vrip_res3.ply";
import { PlyOrientedPoint, PlyParser } from "./ply-parser";
import { Vec3, vec3, vec4 } from "wgpu-matrix";

export interface RbfSample {
    position: Vec3;
    target: number;
}

export enum RbfKernel {
    "linear" = 0,
    "gaussian" = 1,
    "cubic" = 2,
    "quintic" = 3,
    "thinPlate" = 4,
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
    boxMin: Vec3;
    boxMax: Vec3;
}

export function createBuiltInRbfFit(
    config: RbfFitConfig,
    sceneId: string,
): RbfFitResult {
    let samples: RbfSample[];
    switch (sceneId) {
        case "torus":
            samples = createTorusConstraintSamples(config);
            break;
        case "dragon":
            samples = createPlyObjectConstraintSamples(config, dragonPly);
            break;
        case "bunny":
            samples = createPlyObjectConstraintSamples(config, bunnyPly);
            break;
        case "sphere":
        default:
            samples = createSphereConstraintSamples(config);
            break;
    }
    const targets = new Float32Array(samples.length);
    const positions = new Float32Array(samples.length * 4);

    for (let index = 0; index < samples.length; index += 1) {
        const sample = samples[index];
        targets[index] = sample.target;

        const pos = vec4.fromValues(...sample.position, 0.0);
        positions.set(pos, index * 4);
    }

    const start: number = performance.now();
    const weights: Float32Array<ArrayBuffer> = solveRbfWeights(samples, config);
    const end: number = performance.now();
    console.log(`Linear system solve took: ${end - start} ms`);

    let positionsArray: Vec3[] = samples.map((sample) => sample.position);
    let { boxMin, boxMax } = createBoundingVolumeFromPoints(
        positionsArray,
        config.normalOffset,
    );

    return {
        samples,
        positions,
        targets,
        weights,
        boxMin,
        boxMax,
    };
}

function createBoundingVolumeFromPoints(
    points: Vec3[],
    padding: number,
): { boxMin: Vec3; boxMax: Vec3 } {
    let boxMin: Vec3 = vec3.create(Infinity, Infinity, Infinity);
    let boxMax: Vec3 = vec3.create(-Infinity, -Infinity, -Infinity);

    for (let i = 0; i < points.length; i++) {
        vec3.min(points[i], boxMin, boxMin);
        vec3.max(points[i], boxMax, boxMax);
    }

    vec3.add(boxMin, [-padding, -padding, -padding], boxMin);
    vec3.add(boxMax, [padding, padding, padding], boxMax);

    return { boxMin, boxMax };
}

function createPlyObjectConstraintSamples(
    config: RbfFitConfig,
    plyFile: string,
): RbfSample[] {
    const orientedPoints: PlyOrientedPoint[] =
        PlyParser.extractPositionsAndNormals(plyFile, config.sphereRadius * 2);

    return selectPlySamplesGreedy(
        orientedPoints,
        config.surfaceSampleCount,
        config,
    );
}

function createPlyConstraintSamples(
    orientedPoints: PlyOrientedPoint[],
    config: RbfFitConfig,
): RbfSample[] {
    return orientedPoints.flatMap((point) =>
        createPlyPointConstraintSamples(point, config),
    );
}

function createPlyPointConstraintSamples(
    point: PlyOrientedPoint,
    config: RbfFitConfig,
): RbfSample[] {
    const samples: RbfSample[] = [{ position: point.position, target: 0 }];

    if (config.normalOffset > 0) {
        const offset = config.normalOffset;
        samples.push({
            position: vec3.create(
                point.position[0] + offset * point.normal[0],
                point.position[1] + offset * point.normal[1],
                point.position[2] + offset * point.normal[2],
            ),
            target: offset,
        });
    }

    return samples;
}

function selectPlySamplesGreedy(
    allPoints: PlyOrientedPoint[],
    maxPoints: number,
    config: RbfFitConfig,
): RbfSample[] {
    const pointLimit = Math.min(Math.floor(maxPoints), allPoints.length);
    if (pointLimit <= 0 || pointLimit === allPoints.length) {
        return createPlyConstraintSamples(
            allPoints.slice(0, pointLimit),
            config,
        );
    }

    const profilingStart = performance.now();

    const initialPointCount = Math.min(32, pointLimit);
    const samplesByPoint = allPoints.map((point) =>
        createPlyPointConstraintSamples(point, config),
    );
    const selectedPointIndices: number[] = [];
    const selectedIndices = new Set<number>();
    const initialStep = allPoints.length / initialPointCount;

    for (let index = 0; index < initialPointCount; index += 1) {
        const pointIndex = Math.floor(index * initialStep);
        selectedIndices.add(pointIndex);
        selectedPointIndices.push(pointIndex);
    }

    while (selectedPointIndices.length < pointLimit) {
        const iterationStart = performance.now();
        const selectedSamples = selectedPointIndices.flatMap(
            (index) => samplesByPoint[index],
        );
        const weights = solveRbfWeights(selectedSamples, config);
        let nextPointIndex: number | undefined;
        let largestResidual = -Infinity;

        for (let index = 0; index < allPoints.length; index += 1) {
            if (selectedIndices.has(index)) {
                continue;
            }

            let residual = 0;
            for (const sample of samplesByPoint[index]) {
                const value = evaluateRbfAt(
                    sample.position,
                    selectedSamples,
                    weights,
                    config,
                );
                residual = Math.max(residual, Math.abs(sample.target - value));
            }

            if (residual > largestResidual) {
                largestResidual = residual;
                nextPointIndex = index;
            }
        }

        if (nextPointIndex === undefined) {
            break;
        }

        selectedIndices.add(nextPointIndex);
        selectedPointIndices.push(nextPointIndex);

        console.log(
            `[Greedy] points=${selectedPointIndices.length} ` +
                `iteration=${((performance.now() - iterationStart) / 1000).toFixed(1)}s ` +
                `heap=${getUsedHeapSize()}`,
        );
    }

    console.log(
        `[Greedy] completed points=${selectedPointIndices.length} ` +
            `elapsed=${((performance.now() - profilingStart) / 1000 / 60).toFixed(1)}min ` +
            `heap=${getUsedHeapSize()}`,
    );

    return selectedPointIndices.flatMap((index) => samplesByPoint[index]);
}

function getUsedHeapSize(): string {
    const memory = (
        performance as Performance & {
            memory?: { usedJSHeapSize: number };
        }
    ).memory;

    if (!memory) {
        return "unavailable";
    }

    return `${(memory.usedJSHeapSize / (1024 * 1024)).toFixed(1)}MB`;
}

function evaluateRbfAt(
    point: Vec3,
    samples: RbfSample[],
    weights: Float32Array<ArrayBuffer>,
    config: RbfFitConfig,
): number {
    let value = 0;

    for (let index = 0; index < samples.length; index += 1) {
        const radius = vec3.distance(point, samples[index].position);
        value += weights[index] * kernel(radius, config);
    }

    return value;
}

function createSphereConstraintSamples(config: RbfFitConfig): RbfSample[] {
    const samples: RbfSample[] = [];
    const surfacePoints: [number, number, number][] = [];
    const radius = config.sphereRadius;

    for (let index = 0; index < config.surfaceSampleCount; index += 1) {
        const direction = fibonacciDirection(index, config.surfaceSampleCount);
        const surfacePoint = vec3.scale(direction, radius);

        surfacePoints.push([surfacePoint[0], surfacePoint[1], surfacePoint[2]]);
        samples.push({
            position: vec3.create(
                surfacePoint[0],
                surfacePoint[1],
                surfacePoint[2],
            ),
            target: 0,
        });

        if (config.normalOffset > 0) {
            samples.push({
                position: vec3.create(
                    surfacePoint[0] + direction[0] * config.normalOffset,
                    surfacePoint[1] + direction[1] * config.normalOffset,
                    surfacePoint[2] + direction[2] * config.normalOffset,
                ),
                target: config.normalOffset,
            });

            // samples.push({
            //     position: [
            //         surfacePoint[0] - direction[0] * config.normalOffset,
            //         surfacePoint[1] - direction[1] * config.normalOffset,
            //         surfacePoint[2] - direction[2] * config.normalOffset,
            //     ],
            //     target: -config.normalOffset,
            // });
        }
    }

    // Ponto âncora interno: Garante que o campo RBF fique negativo no núcleo da esfera
    //samples.push({ position: [0, 0, 0], target: -radius });

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

function createTorusConstraintSamples(config: RbfFitConfig): RbfSample[] {
    const samples: RbfSample[] = [];
    const surfacePoints: [number, number, number][] = [];
    const majorRadius = config.sphereRadius;
    const minorRadius = majorRadius * 0.4;

    const sampleCount = config.surfaceSampleCount;
    const numU = Math.max(2, Math.floor(Math.sqrt(sampleCount * 2)));
    const numV = Math.max(2, Math.floor(sampleCount / numU) + 1);

    for (let i = 0; i < numU; i++) {
        for (let j = 0; j < numV; j++) {
            //if (samples.length >= sampleCount * 3) break;
            const u = (i / numU) * Math.PI * 2;
            const v = (j / numV) * Math.PI * 2;

            const x = (majorRadius + minorRadius * Math.cos(v)) * Math.cos(u);
            const z = (majorRadius + minorRadius * Math.cos(v)) * Math.sin(u);
            const y = minorRadius * Math.sin(v);

            const surfacePoint: [number, number, number] = [x, y, z];
            surfacePoints.push(surfacePoint);
            samples.push({
                position: vec3.create(
                    surfacePoint[0],
                    surfacePoint[1],
                    surfacePoint[2],
                ),
                target: 0,
            });

            if (config.normalOffset > 0) {
                const nx = Math.cos(v) * Math.cos(u);
                const nz = Math.cos(v) * Math.sin(u);
                const ny = Math.sin(v);

                samples.push({
                    position: vec3.create(
                        x + nx * config.normalOffset,
                        y + ny * config.normalOffset,
                        z + nz * config.normalOffset,
                    ),
                    target: config.normalOffset,
                });

                samples.push({
                    position: vec3.create(
                        x - nx * config.normalOffset,
                        y - ny * config.normalOffset,
                        z - nz * config.normalOffset,
                    ),
                    target: -config.normalOffset,
                });
            }
        }
    }

    // Anchor point in the center hole
    // samples.push({
    //     position: vec3.create(0, 0, 0),
    //     target: majorRadius - minorRadius,
    // });
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
            const radius = vec3.distance(
                sampleRow.position,
                sampleColumn.position,
            );
            const diagonal = row === column ? config.regularization : 0;

            matrixRow.push(kernel(radius, config) + diagonal);
        }
        M.push(matrixRow);
    }

    const LU = numeric.LU(M, true);
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
        case RbfKernel.thinPlate:
            if (radius == 0) {
                return 0.0;
            }
            return Math.pow(radius, 2) * Math.log(radius);
        default:
            return radius;
    }
}
