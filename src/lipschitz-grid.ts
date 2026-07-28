import { Vec3, vec3 } from "wgpu-matrix";
import { RbfFitConfig, RbfFitResult } from "./rbf";
import { evaluateKernelDerivativeScale } from "./rbf-field";

export interface LipschitzGridConfig {
    resolution: number;
    samplesPerAxis: number;
    safetyFactor: number;
}

export interface LipschitzGrid {
    nx: number;
    ny: number;
    nz: number;
    boxMin: Vec3;
    boxMax: Vec3;
    cellSize: Vec3;
    values: Float32Array<ArrayBuffer>;
    gradientSampleCount: number;
    minimumBound: number;
    maximumBound: number;
}

const MINIMUM_LIPSCHITZ_BOUND = 1e-4;

export function evaluateRbfGradient(
    point: Vec3,
    fit: RbfFitResult,
    config: RbfFitConfig,
): Vec3 {
    let gradientX = 0;
    let gradientY = 0;
    let gradientZ = 0;

    for (let index = 0; index < fit.samples.length; index += 1) {
        const center = fit.samples[index].position;
        const dx = point[0] - center[0];
        const dy = point[1] - center[1];
        const dz = point[2] - center[2];
        const radiusSquared = dx * dx + dy * dy + dz * dz;
        const radius = Math.sqrt(radiusSquared);
        const derivativeScale = evaluateKernelDerivativeScale(
            radius,
            config.kernel,
            config.gaussianEpsilon,
        );

        const weightedScale = fit.weights[index] * derivativeScale;
        gradientX += weightedScale * dx;
        gradientY += weightedScale * dy;
        gradientZ += weightedScale * dz;
    }

    return vec3.create(gradientX, gradientY, gradientZ);
}

export function buildLipschitzGrid(
    fit: RbfFitResult,
    rbfConfig: RbfFitConfig,
    gridConfig: LipschitzGridConfig,
): LipschitzGrid {
    const resolution = Math.max(1, Math.floor(gridConfig.resolution));
    const samplesPerAxis = Math.max(
        2,
        Math.floor(gridConfig.samplesPerAxis),
    );
    const safetyFactor = Math.max(1, gridConfig.safetyFactor);
    const latticeIntervalsPerCell = samplesPerAxis - 1;
    const latticeResolution =
        resolution * latticeIntervalsPerCell + 1;
    const gradientSampleCount =
        latticeResolution * latticeResolution * latticeResolution;
    const gradientMagnitudes = new Float32Array(gradientSampleCount);
    const boxMin = vec3.create(...fit.boxMin);
    const boxMax = vec3.create(...fit.boxMax);
    const boxSize = vec3.subtract(boxMax, boxMin);
    const cellSize = vec3.scale(boxSize, 1 / resolution);

    for (let z = 0; z < latticeResolution; z += 1) {
        const zRatio = z / (latticeResolution - 1);
        for (let y = 0; y < latticeResolution; y += 1) {
            const yRatio = y / (latticeResolution - 1);
            for (let x = 0; x < latticeResolution; x += 1) {
                const xRatio = x / (latticeResolution - 1);
                const point = vec3.create(
                    boxMin[0] + boxSize[0] * xRatio,
                    boxMin[1] + boxSize[1] * yRatio,
                    boxMin[2] + boxSize[2] * zRatio,
                );
                const gradient = evaluateRbfGradient(
                    point,
                    fit,
                    rbfConfig,
                );
                const latticeIndex =
                    z * latticeResolution * latticeResolution +
                    y * latticeResolution +
                    x;
                gradientMagnitudes[latticeIndex] = vec3.length(gradient);
            }
        }
    }

    const cellCount = resolution * resolution * resolution;
    const rawBounds = new Float32Array(cellCount);
    for (let cellZ = 0; cellZ < resolution; cellZ += 1) {
        for (let cellY = 0; cellY < resolution; cellY += 1) {
            for (let cellX = 0; cellX < resolution; cellX += 1) {
                let maximumGradient = 0;
                for (
                    let sampleZ = 0;
                    sampleZ < samplesPerAxis;
                    sampleZ += 1
                ) {
                    for (
                        let sampleY = 0;
                        sampleY < samplesPerAxis;
                        sampleY += 1
                    ) {
                        for (
                            let sampleX = 0;
                            sampleX < samplesPerAxis;
                            sampleX += 1
                        ) {
                            const latticeX =
                                cellX * latticeIntervalsPerCell + sampleX;
                            const latticeY =
                                cellY * latticeIntervalsPerCell + sampleY;
                            const latticeZ =
                                cellZ * latticeIntervalsPerCell + sampleZ;
                            const latticeIndex =
                                latticeZ *
                                    latticeResolution *
                                    latticeResolution +
                                latticeY * latticeResolution +
                                latticeX;
                            maximumGradient = Math.max(
                                maximumGradient,
                                gradientMagnitudes[latticeIndex],
                            );
                        }
                    }
                }

                rawBounds[
                    flattenGridIndex(
                        cellX,
                        cellY,
                        cellZ,
                        resolution,
                        resolution,
                    )
                ] = maximumGradient;
            }
        }
    }

    const values = new Float32Array(cellCount);
    let minimumBound = Infinity;
    let maximumBound = 0;
    for (let cellZ = 0; cellZ < resolution; cellZ += 1) {
        for (let cellY = 0; cellY < resolution; cellY += 1) {
            for (let cellX = 0; cellX < resolution; cellX += 1) {
                let neighborhoodMaximum = 0;
                for (let offsetZ = -1; offsetZ <= 1; offsetZ += 1) {
                    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
                        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
                            const neighborX = cellX + offsetX;
                            const neighborY = cellY + offsetY;
                            const neighborZ = cellZ + offsetZ;
                            if (
                                neighborX < 0 ||
                                neighborX >= resolution ||
                                neighborY < 0 ||
                                neighborY >= resolution ||
                                neighborZ < 0 ||
                                neighborZ >= resolution
                            ) {
                                continue;
                            }

                            neighborhoodMaximum = Math.max(
                                neighborhoodMaximum,
                                rawBounds[
                                    flattenGridIndex(
                                        neighborX,
                                        neighborY,
                                        neighborZ,
                                        resolution,
                                        resolution,
                                    )
                                ],
                            );
                        }
                    }
                }

                const bound = Math.max(
                    neighborhoodMaximum * safetyFactor,
                    MINIMUM_LIPSCHITZ_BOUND,
                );
                const cellIndex = flattenGridIndex(
                    cellX,
                    cellY,
                    cellZ,
                    resolution,
                    resolution,
                );
                values[cellIndex] = bound;
                minimumBound = Math.min(minimumBound, bound);
                maximumBound = Math.max(maximumBound, bound);
            }
        }
    }

    return {
        nx: resolution,
        ny: resolution,
        nz: resolution,
        boxMin,
        boxMax,
        cellSize,
        values,
        gradientSampleCount,
        minimumBound,
        maximumBound,
    };
}

export function flattenGridIndex(
    x: number,
    y: number,
    z: number,
    nx: number,
    ny: number,
): number {
    return z * nx * ny + y * nx + x;
}
