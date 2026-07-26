import { Vec3 } from "wgpu-matrix";
import { RbfFitConfig, RbfFitResult } from "./rbf";
import { extractMarchingCubesMesh, sampleScalarGrid } from "./marching-cubes";

export interface ExtractedMesh {
    positions: Float32Array<ArrayBuffer>;
    normals: Float32Array<ArrayBuffer>;
    indices?: Uint32Array;
}

export interface MeshBounds {
    min: Vec3;
    max: Vec3;
}

export function createExtractedMesh(
    positions: number[],
    normals: number[],
): ExtractedMesh {
    return {
        positions: new Float32Array(positions),
        normals: new Float32Array(normals),
    };
}

export function buildMarchingCubesMesh(
    rbfFitResult: RbfFitResult,
    rbfFitConfig: RbfFitConfig,
    resolution: number,
    isoValue: number,
    extraPadding: number,
): ExtractedMesh {
    const scalarGrid = sampleScalarGrid(
        rbfFitResult,
        rbfFitConfig,
        resolution,
        extraPadding,
    );
    const rawMesh = extractMarchingCubesMesh(scalarGrid, isoValue);

    return createExtractedMesh(rawMesh.positions, rawMesh.normals);
}
