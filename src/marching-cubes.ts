import { vec3, Vec3 } from "wgpu-matrix";
import { RbfFitConfig, RbfFitResult } from "./rbf";
import { evaluateRbfField } from "./rbf-field";
import { edgeTable, triTable } from "./marching-cubes-tables";
import { createNoise3D } from "simplex-noise";

export interface ScalarGrid {
    nx: number;
    ny: number;
    nz: number;
    boxMin: Vec3;
    boxMax: Vec3;
    step: Vec3;
    values: Float32Array;
}

export function indexFromCoord(
    x: number,
    y: number,
    z: number,
    nx: number,
    ny: number,
    nz: number,
) {
    return z * nx * ny + y * nx + x;
}

function scalarField(x: number, y: number, z: number): number {
    const noiseScale = 0.1;
    const noise = createNoise3D();
    return noise(x * 0.1, y * 0.1, z * 0.1) - y * 0.5;
}

export function sampleScalarGrid(
    rbfFitResult: RbfFitResult,
    rbfFitConfig: RbfFitConfig,
    resolution: number,
    extraPadding: number,
): ScalarGrid {
    let voxelGrid: number[] = [];
    const nx = resolution;
    const ny = resolution;
    const nz = resolution;

    const paddedBoxMin: Vec3 = vec3.add(
        rbfFitResult.boxMin,
        vec3.create(-extraPadding, -extraPadding, -extraPadding),
    );
    const paddedBoxMax: Vec3 = vec3.add(
        rbfFitResult.boxMax,
        vec3.create(extraPadding, extraPadding, extraPadding),
    );
    const step: Vec3 = vec3.create(
        (paddedBoxMax[0] - paddedBoxMin[0]) / (nx - 1),
        (paddedBoxMax[1] - paddedBoxMin[1]) / (ny - 1),
        (paddedBoxMax[2] - paddedBoxMin[2]) / (nz - 1),
    );

    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const gridPoint: Vec3 = vec3.create(
                    paddedBoxMin[0] + x * step[0],
                    paddedBoxMin[1] + y * step[1],
                    paddedBoxMin[2] + z * step[2],
                );

                voxelGrid.push(
                    evaluateRbfField(gridPoint, rbfFitResult, rbfFitConfig),
                );
                // const s = scalarField(gridPoint[0], gridPoint[1], gridPoint[2]);
                // voxelGrid.push(s);
            }
        }
    }

    return {
        boxMax: paddedBoxMax,
        boxMin: paddedBoxMin,
        nx: nx,
        ny: ny,
        nz: nz,
        step: step,
        values: new Float32Array(voxelGrid),
    };
}

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
export function VertexInterp(
    isolevel: number,
    p1: Vec3,
    p2: Vec3,
    valP1: number,
    valP2: number,
): Vec3 {
    let mu: number;
    let p: Vec3 = vec3.create();

    if (Math.abs(isolevel - valP1) < 0.00001) return p1;
    if (Math.abs(isolevel - valP2) < 0.00001) return p2;
    if (Math.abs(valP1 - valP2) < 0.00001) return p1;
    mu = (isolevel - valP1) / (valP2 - valP1);
    p[0] = p1[0] + mu * (p2[0] - p1[0]);
    p[1] = p1[1] + mu * (p2[1] - p1[1]);
    p[2] = p1[2] + mu * (p2[2] - p1[2]);

    return p;
}

function gridPointPosition(
    grid: ScalarGrid,
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

export interface RawMesh {
    positions: number[];
    normals: number[];
}

export function extractMarchingCubesMesh(
    grid: ScalarGrid,
    isoValue: number,
): RawMesh {
    const positions: number[] = [];
    const normals: number[] = [];

    for (let z = 0; z < grid.nz - 1; z++) {
        for (let y = 0; y < grid.ny - 1; y++) {
            for (let x = 0; x < grid.nx - 1; x++) {
                const cornerCoords = [
                    [x, y, z],
                    [x + 1, y, z],
                    [x + 1, y + 1, z],
                    [x, y + 1, z],
                    [x, y, z + 1],
                    [x + 1, y, z + 1],
                    [x + 1, y + 1, z + 1],
                    [x, y + 1, z + 1],
                ];

                const cornerValues: number[] = [];
                const cornerPos: Vec3[] = [];

                for (let [cx, cy, cz] of cornerCoords) {
                    const index = indexFromCoord(
                        cx,
                        cy,
                        cz,
                        grid.nx,
                        grid.ny,
                        grid.nz,
                    );

                    cornerValues.push(grid.values[index]);
                    cornerPos.push(gridPointPosition(grid, cx, cy, cz));
                }

                /*
                    Determine the index into the edge table which
                    tells us which vertices are inside of the surface
                */
                let cubeindex = 0;
                if (cornerValues[0] < isoValue) cubeindex |= 1;
                if (cornerValues[1] < isoValue) cubeindex |= 2;
                if (cornerValues[2] < isoValue) cubeindex |= 4;
                if (cornerValues[3] < isoValue) cubeindex |= 8;
                if (cornerValues[4] < isoValue) cubeindex |= 16;
                if (cornerValues[5] < isoValue) cubeindex |= 32;
                if (cornerValues[6] < isoValue) cubeindex |= 64;
                if (cornerValues[7] < isoValue) cubeindex |= 128;

                /* Cube is entirely in/out of the surface */
                if (edgeTable[cubeindex] == 0) {
                    continue;
                }

                const vertList: Vec3[] = new Array(12);

                /* Find the vertices where the surface intersects the cube */
                if (edgeTable[cubeindex] & 1)
                    vertList[0] = VertexInterp(
                        isoValue,
                        cornerPos[0],
                        cornerPos[1],
                        cornerValues[0],
                        cornerValues[1],
                    );
                if (edgeTable[cubeindex] & 2)
                    vertList[1] = VertexInterp(
                        isoValue,
                        cornerPos[1],
                        cornerPos[2],
                        cornerValues[1],
                        cornerValues[2],
                    );
                if (edgeTable[cubeindex] & 4)
                    vertList[2] = VertexInterp(
                        isoValue,
                        cornerPos[2],
                        cornerPos[3],
                        cornerValues[2],
                        cornerValues[3],
                    );
                if (edgeTable[cubeindex] & 8)
                    vertList[3] = VertexInterp(
                        isoValue,
                        cornerPos[3],
                        cornerPos[0],
                        cornerValues[3],
                        cornerValues[0],
                    );
                if (edgeTable[cubeindex] & 16)
                    vertList[4] = VertexInterp(
                        isoValue,
                        cornerPos[4],
                        cornerPos[5],
                        cornerValues[4],
                        cornerValues[5],
                    );
                if (edgeTable[cubeindex] & 32)
                    vertList[5] = VertexInterp(
                        isoValue,
                        cornerPos[5],
                        cornerPos[6],
                        cornerValues[5],
                        cornerValues[6],
                    );
                if (edgeTable[cubeindex] & 64)
                    vertList[6] = VertexInterp(
                        isoValue,
                        cornerPos[6],
                        cornerPos[7],
                        cornerValues[6],
                        cornerValues[7],
                    );
                if (edgeTable[cubeindex] & 128)
                    vertList[7] = VertexInterp(
                        isoValue,
                        cornerPos[7],
                        cornerPos[4],
                        cornerValues[7],
                        cornerValues[4],
                    );
                if (edgeTable[cubeindex] & 256)
                    vertList[8] = VertexInterp(
                        isoValue,
                        cornerPos[0],
                        cornerPos[4],
                        cornerValues[0],
                        cornerValues[4],
                    );
                if (edgeTable[cubeindex] & 512)
                    vertList[9] = VertexInterp(
                        isoValue,
                        cornerPos[1],
                        cornerPos[5],
                        cornerValues[1],
                        cornerValues[5],
                    );
                if (edgeTable[cubeindex] & 1024)
                    vertList[10] = VertexInterp(
                        isoValue,
                        cornerPos[2],
                        cornerPos[6],
                        cornerValues[2],
                        cornerValues[6],
                    );
                if (edgeTable[cubeindex] & 2048)
                    vertList[11] = VertexInterp(
                        isoValue,
                        cornerPos[3],
                        cornerPos[7],
                        cornerValues[3],
                        cornerValues[7],
                    );
                //const triangles: Triangle[] = [];
                //let nTriangles = 0;
                for (let i = 0; triTable[cubeindex][i] != -1; i += 3) {
                    const v0 = vertList[triTable[cubeindex][i]];
                    const v1 = vertList[triTable[cubeindex][i + 1]];
                    const v2 = vertList[triTable[cubeindex][i + 2]];

                    positions.push(v0[0], v0[1], v0[2]);
                    positions.push(v1[0], v1[1], v1[2]);
                    positions.push(v2[0], v2[1], v2[2]);

                    // compute normal
                    const edgeA = vec3.sub(v1, v0);
                    const edgeB = vec3.sub(v2, v0);
                    const normal = vec3.normalize(vec3.cross(edgeA, edgeB));

                    // duplicate normals for all 3 points in triangle
                    normals.push(normal[0], normal[1], normal[2]);
                    normals.push(normal[0], normal[1], normal[2]);
                    normals.push(normal[0], normal[1], normal[2]);
                }
            }
        }
    }
    return { positions: positions, normals: normals };
}

interface Triangle {
    vertices: Vec3[];
}
