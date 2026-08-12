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
    const gradientSampleCount =
        resolution * resolution * resolution *
        samplesPerAxis * samplesPerAxis * samplesPerAxis;
    const boxMin = vec3.create(...fit.boxMin);
    const boxMax = vec3.create(...fit.boxMax);
    const boxSize = vec3.subtract(boxMax, boxMin);
    const cellSize = vec3.scale(boxSize, 1 / resolution);

    const cellCount = resolution * resolution * resolution;
    const values = new Float32Array(cellCount);
    let minimumBound = Infinity;
    let maximumBound = 0;
    for (let cellZ = 0; cellZ < resolution; cellZ += 1) {
        for (let cellY = 0; cellY < resolution; cellY += 1) {
            for (let cellX = 0; cellX < resolution; cellX += 1) {
                let maximumGradient = 0;
                for (let sampleZ = 0; sampleZ < samplesPerAxis; sampleZ += 1) {
                    const zRatio = sampleZ / (samplesPerAxis - 1);
                    for (let sampleY = 0; sampleY < samplesPerAxis; sampleY += 1) {
                        const yRatio = sampleY / (samplesPerAxis - 1);
                        for (let sampleX = 0; sampleX < samplesPerAxis; sampleX += 1) {
                            const xRatio = sampleX / (samplesPerAxis - 1);
                            const point = vec3.create(
                                boxMin[0] + (cellX + xRatio) * cellSize[0],
                                boxMin[1] + (cellY + yRatio) * cellSize[1],
                                boxMin[2] + (cellZ + zRatio) * cellSize[2],
                            );
                            maximumGradient = Math.max(
                                maximumGradient,
                                vec3.length(
                                    evaluateRbfGradient(
                                        point,
                                        fit,
                                        rbfConfig,
                                    ),
                                ),
                            );
                        }
                    }
                }
                const bound = Math.max(
                    maximumGradient * safetyFactor,
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
