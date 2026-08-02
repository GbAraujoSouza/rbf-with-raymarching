import { Vec3, vec3 } from "wgpu-matrix";

export interface PlyOrientedPoint {
    position: Vec3;
    normal: Vec3;
}

export interface PlyParserOptions {
    normalize?: boolean;
    targetSize?: number;
    maxPoints?: number;
    flipNormalsOutward?: boolean;
}

interface PlyHeader {
    vertexCount: number;
    faceCount: number;
    headerEndLine: number;
    vertexProperties: string[];
}

export class PlyParser {
    static extractPositionsAndNormals(
        plyFile: string,
        options: PlyParserOptions = {},
    ): PlyOrientedPoint[] {
        const lines = plyFile.split("\n");
        const header = this.parseHeader(lines);
        const vertices = this.parseVertices(lines, header);
        const normals = this.calculateVertexNormals(
            lines,
            header,
            vertices,
            options.flipNormalsOutward ?? true,
        );
        const normalizedVertices =
            (options.normalize ?? true)
                ? this.normalizeVertices(vertices, options.targetSize ?? 2.0)
                : vertices;
        const orientedPoints = this.createOrientedPoints(
            normalizedVertices,
            normals,
        );

        return this.downsample(orientedPoints, options.maxPoints);
    }

    private static parseHeader(lines: string[]): PlyHeader {
        if (lines[0]?.trim() !== "ply") {
            throw new Error("Invalid PLY file: missing ply header");
        }

        let vertexCount = 0;
        let faceCount = 0;
        let headerEndLine = -1;
        let isAscii = false;
        let currentElement = "";
        const vertexProperties: string[] = [];

        for (let lineIndex = 1; lineIndex < lines.length; lineIndex += 1) {
            const line = lines[lineIndex].trim();
            const parts = line.split(/\s+/);

            if (line === "format ascii 1.0") {
                isAscii = true;
                continue;
            }

            if (parts[0] === "format") {
                throw new Error(`Unsupported PLY format: ${line}`);
            }

            if (parts[0] === "element") {
                currentElement = parts[1];
                if (currentElement === "vertex") {
                    vertexCount = Number.parseInt(parts[2], 10);
                }
                if (currentElement === "face") {
                    faceCount = Number.parseInt(parts[2], 10);
                }
                continue;
            }

            if (parts[0] === "property" && currentElement === "vertex") {
                vertexProperties.push(parts[parts.length - 1]);
                continue;
            }

            if (line === "end_header") {
                headerEndLine = lineIndex;
                break;
            }
        }

        if (headerEndLine < 0 || vertexCount <= 0) {
            throw new Error("Invalid PLY file: incomplete header");
        }

        if (!isAscii) {
            throw new Error("Unsupported PLY format: expected ascii 1.0");
        }

        return {
            vertexCount,
            faceCount,
            headerEndLine,
            vertexProperties,
        };
    }

    private static parseVertices(lines: string[], header: PlyHeader): Vec3[] {
        const xIndex = header.vertexProperties.indexOf("x");
        const yIndex = header.vertexProperties.indexOf("y");
        const zIndex = header.vertexProperties.indexOf("z");

        const vertices: Vec3[] = [];
        const firstVertexLine = header.headerEndLine + 1;

        for (let index = 0; index < header.vertexCount; index += 1) {
            const line = lines[firstVertexLine + index]?.trim();

            const values: number[] = line
                .split(/\s+/)
                .map((value) => Number.parseFloat(value));
            const x: number = values[xIndex];
            const y: number = values[yIndex];
            const z: number = values[zIndex];

            vertices.push(vec3.create(x, y, z));
        }

        return vertices;
    }

    private static calculateVertexNormals(
        lines: string[],
        header: PlyHeader,
        vertices: Vec3[],
        flipNormalsOutward: boolean,
    ): Vec3[] {
        const normals = vertices.map(() => vec3.create(0, 0, 0));
        const firstFaceLine = header.headerEndLine + header.vertexCount + 1;
        let signedVolume = 0;

        if (header.faceCount > 0) {
            signedVolume = this.accumulateFaceNormals(
                lines,
                header,
                vertices,
                normals,
                firstFaceLine,
            );
        }

        if (flipNormalsOutward && signedVolume < 0) {
            for (const normal of normals) {
                vec3.scale(normal, -1, normal);
            }
        }

        for (const normal of normals) {
            if (vec3.length(normal) === 0) {
                normal[1] = 1;
            } else {
                vec3.normalize(normal, normal);
            }
        }

        return normals;
    }

    private static accumulateFaceNormals(
        lines: string[],
        header: PlyHeader,
        vertices: Vec3[],
        normals: Vec3[],
        firstFaceLine: number,
    ): number {
        let signedVolume = 0;

        for (let faceIndex = 0; faceIndex < header.faceCount; faceIndex += 1) {
            const line = lines[firstFaceLine + faceIndex]?.trim();

            const indices: number[] = this.parseFaceIndices(line, faceIndex);

            signedVolume += this.accumulateTriangleNormal(
                vertices,
                normals,
                indices[0],
                indices[1],
                indices[2],
            );
        }

        return signedVolume;
    }

    private static accumulateTriangleNormal(
        vertices: Vec3[],
        normals: Vec3[],
        aIndex: number,
        bIndex: number,
        cIndex: number,
    ): number {
        const a = vertices[aIndex];
        const b = vertices[bIndex];
        const c = vertices[cIndex];

        const ab = vec3.subtract(b, a);
        const ac = vec3.subtract(c, a);
        const faceNormal = vec3.cross(ab, ac);

        vec3.add(normals[aIndex], faceNormal, normals[aIndex]);
        vec3.add(normals[bIndex], faceNormal, normals[bIndex]);
        vec3.add(normals[cIndex], faceNormal, normals[cIndex]);

        return vec3.dot(a, vec3.cross(b, c)) / 6;
    }

    private static parseFaceIndices(line: string, faceIndex: number): number[] {
        const values = line
            .split(/\s+/)
            .map((value) => Number.parseInt(value, 10));
        const vertexCount = values[0];
        const indices = values.slice(1, vertexCount + 1);

        return indices;
    }

    private static normalizeVertices(
        vertices: Vec3[],
        targetSize: number,
    ): Vec3[] {
        const boxMin = vec3.create(Infinity, Infinity, Infinity);
        const boxMax = vec3.create(-Infinity, -Infinity, -Infinity);

        for (const vertex of vertices) {
            vec3.min(vertex, boxMin, boxMin);
            vec3.max(vertex, boxMax, boxMax);
        }

        const center = vec3.scale(vec3.add(boxMin, boxMax), 0.5);
        const size = vec3.subtract(boxMax, boxMin);
        const largestAxis = Math.max(size[0], size[1], size[2]);
        const scale = largestAxis > 0 ? targetSize / largestAxis : 1;

        return vertices.map((vertex) =>
            vec3.scale(vec3.subtract(vertex, center), scale),
        );
    }

    private static createOrientedPoints(
        vertices: Vec3[],
        normals: Vec3[],
    ): PlyOrientedPoint[] {
        const orientedPoints: PlyOrientedPoint[] = [];

        for (let index = 0; index < vertices.length; index += 1) {
            const position = vertices[index];
            const normal = vec3.copy(normals[index]);

            orientedPoints.push({ position, normal });
        }

        return orientedPoints;
    }

    private static downsample(
        points: PlyOrientedPoint[],
        maxPoints?: number,
    ): PlyOrientedPoint[] {
        if (!maxPoints || maxPoints <= 0 || points.length <= maxPoints) {
            return points;
        }

        const result: PlyOrientedPoint[] = [];
        const step = points.length / maxPoints;

        for (let index = 0; index < maxPoints; index += 1) {
            result.push(points[Math.floor(index * step)]);
        }

        return result;
    }
}
