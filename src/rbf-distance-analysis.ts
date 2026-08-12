import { Vec3, vec3 } from "wgpu-matrix";
import { evaluateRbfField } from "./rbf-field";
import { evaluateRbfGradient } from "./lipschitz-grid";
import { RbfFitConfig, RbfFitResult } from "./rbf";

export const RBF_DISTANCE_GRID_RESOLUTION = 16;

const ZERO_TOLERANCE = 1e-8;

export interface RbfGridSamples {
    resolution: number;
    size: number;
    boxMin: Vec3;
    boxMax: Vec3;
    step: Vec3;
    values: Float32Array;
}

export interface RbfDistanceSample {
    position: Vec3;
    rbf: number;
    dist: number;
    gradientMagnitude: number;
}

export interface RbfDistanceAnalysis {
    samples: RbfDistanceSample[];
    zeroCrossingCount: number;
    gridSampleCount: number;
    elapsedMilliseconds: number;
}

function gridIndex(x: number, y: number, z: number, size: number): number {
    return z * size * size + y * size + x;
}

function gridPosition(
    grid: RbfGridSamples,
    x: number,
    y: number,
    z: number,
): Vec3 {
    return vec3.create(
        grid.boxMin[0] + x * grid.step[0],
        grid.boxMin[1] + y * grid.step[1],
        grid.boxMin[2] + z * grid.step[2],
    );
}

export function sampleBoundingBox(
    fit: RbfFitResult,
    config: RbfFitConfig,
    resolution: number = RBF_DISTANCE_GRID_RESOLUTION,
): RbfGridSamples {
    const size = resolution + 1;
    const step = vec3.create(
        (fit.boxMax[0] - fit.boxMin[0]) / resolution,
        (fit.boxMax[1] - fit.boxMin[1]) / resolution,
        (fit.boxMax[2] - fit.boxMin[2]) / resolution,
    );
    const values = new Float32Array(size * size * size);

    for (let z = 0; z < size; z++) {
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const position = vec3.create(
                    fit.boxMin[0] + x * step[0],
                    fit.boxMin[1] + y * step[1],
                    fit.boxMin[2] + z * step[2],
                );
                values[gridIndex(x, y, z, size)] = evaluateRbfField(
                    position,
                    fit,
                    config,
                );
            }
        }
    }

    return {
        resolution,
        size,
        boxMin: vec3.clone(fit.boxMin),
        boxMax: vec3.clone(fit.boxMax),
        step,
        values,
    };
}

export function findZeroCrossings(grid: RbfGridSamples): Vec3[] {
    const zeroes: Vec3[] = [];
    const seen = new Set<string>();

    const addEdgeCrossing = (
        x0: number,
        y0: number,
        z0: number,
        x1: number,
        y1: number,
        z1: number,
    ): void => {
        const u = grid.values[gridIndex(x0, y0, z0, grid.size)];
        const v = grid.values[gridIndex(x1, y1, z1, grid.size)];

        if (!Number.isFinite(u) || !Number.isFinite(v)) {
            return;
        }

        const uIsZero = Math.abs(u) <= ZERO_TOLERANCE;
        const vIsZero = Math.abs(v) <= ZERO_TOLERANCE;
        if ((!uIsZero && !vIsZero && u < 0 === v < 0) || (uIsZero && vIsZero)) {
            return;
        }

        const p0 = gridPosition(grid, x0, y0, z0);
        const p1 = gridPosition(grid, x1, y1, z1);
        let t = 0;
        if (!uIsZero) {
            t = vIsZero ? 1 : u / (u - v);
        }

        const crossing = vec3.create(
            p0[0] + t * (p1[0] - p0[0]),
            p0[1] + t * (p1[1] - p0[1]),
            p0[2] + t * (p1[2] - p0[2]),
        );
        const key = `${crossing[0].toPrecision(8)},${crossing[1].toPrecision(8)},${crossing[2].toPrecision(8)}`;
        if (!seen.has(key)) {
            seen.add(key);
            zeroes.push(crossing);
        }
    };

    // X-directed edges.
    for (let z = 0; z < grid.size; z++) {
        for (let y = 0; y < grid.size; y++) {
            for (let x = 0; x < grid.resolution; x++) {
                addEdgeCrossing(x, y, z, x + 1, y, z);
            }
        }
    }

    // Y-directed edges.
    for (let z = 0; z < grid.size; z++) {
        for (let y = 0; y < grid.resolution; y++) {
            for (let x = 0; x < grid.size; x++) {
                addEdgeCrossing(x, y, z, x, y + 1, z);
            }
        }
    }

    // Z-directed edges.
    for (let z = 0; z < grid.resolution; z++) {
        for (let y = 0; y < grid.size; y++) {
            for (let x = 0; x < grid.size; x++) {
                addEdgeCrossing(x, y, z, x, y, z + 1);
            }
        }
    }

    return zeroes;
}

export function calculateDistances(
    grid: RbfGridSamples,
    zeroes: Vec3[],
    fit: RbfFitResult,
    config: RbfFitConfig,
): RbfDistanceSample[] {
    if (zeroes.length === 0) {
        return [];
    }

    const distances: RbfDistanceSample[] = [];

    for (let z = 0; z < grid.size; z++) {
        for (let y = 0; y < grid.size; y++) {
            for (let x = 0; x < grid.size; x++) {
                const position = gridPosition(grid, x, y, z);
                const fieldValue = grid.values[gridIndex(x, y, z, grid.size)];

                let minimumSquaredDistance = Number.POSITIVE_INFINITY;
                for (const zero of zeroes) {
                    const dx = zero[0] - position[0];
                    const dy = zero[1] - position[1];
                    const dz = zero[2] - position[2];
                    const squaredDistance = dx * dx + dy * dy + dz * dz;
                    minimumSquaredDistance = Math.min(
                        minimumSquaredDistance,
                        squaredDistance,
                    );
                }

                const dist = Math.sqrt(minimumSquaredDistance);
                if (Number.isFinite(dist)) {
                    distances.push({
                        position,
                        rbf: Math.abs(fieldValue),
                        dist,
                        gradientMagnitude: vec3.length(
                            evaluateRbfGradient(position, fit, config),
                        ),
                    });
                }
            }
        }
    }

    return distances;
}

export function analyzeRbfDistance(
    fit: RbfFitResult,
    config: RbfFitConfig,
    resolution: number = RBF_DISTANCE_GRID_RESOLUTION,
): RbfDistanceAnalysis {
    const start = performance.now();
    const grid = sampleBoundingBox(fit, config, resolution);
    const zeroes = findZeroCrossings(grid);
    const samples = calculateDistances(grid, zeroes, fit, config);

    return {
        samples,
        zeroCrossingCount: zeroes.length,
        gridSampleCount: grid.values.length,
        elapsedMilliseconds: performance.now() - start,
    };
}
