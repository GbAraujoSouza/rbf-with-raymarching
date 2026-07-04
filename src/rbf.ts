// @ts-ignore
import numeric from "numeric";

import dragonObj from "./dragon_points.obj";
import carObj from "./points_block_car.obj";
import { ObjParser } from "./obj-parser";
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
            samples = createObjectConstraintSamples(config);
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
        // positions[index * 4 + 0] = sample.position[0];
        // positions[index * 4 + 1] = sample.position[1];
        // positions[index * 4 + 2] = sample.position[2];
        // positions[index * 4 + 3] = 0;

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

function createObjectConstraintSamples(config: RbfFitConfig): RbfSample[] {
    const samples: RbfSample[] = [];
    const orientedPos = ObjParser.extractPositionsAndNormals(carObj);
    orientedPos.forEach((orientedPoint) => {
        samples.push({
            position: vec3.create(
                orientedPoint.position.x,
                orientedPoint.position.y,
                orientedPoint.position.z,
            ),
            target: 0,
        });
        if (config.normalOffset > 0) {
            const offset: number = config.normalOffset;
            samples.push({
                position: vec3.create(
                    orientedPoint.position.x + offset * orientedPoint.normal.x,
                    orientedPoint.position.y + offset * orientedPoint.normal.y,
                    orientedPoint.position.z + offset * orientedPoint.normal.z,
                ),
                target: offset,
            });
        }
    });

    // const bounds = 1.0;
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
    //     for (const sp of orientedPos) {
    //         // kernel ou distancia
    //         const dist = distance(anchor, sp.position as any);
    //         minDist = Math.min(minDist, dist);
    //     }
    //     samples.push({ position: anchor, target: minDist });
    // }

    return samples;
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
            if (samples.length >= sampleCount * 3) break;
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

                // const bounds = majorRadius * 2.0;
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
            }
        }
    }

    // Anchor point in the center hole
    samples.push({
        position: vec3.create(0, 0, 0),
        target: majorRadius - minorRadius,
    });
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
