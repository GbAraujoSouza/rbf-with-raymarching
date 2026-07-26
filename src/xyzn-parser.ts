import { Vec3, vec3 } from "wgpu-matrix";

export interface XyznPoint {
    position: Vec3;
    normal: Vec3;
}

export class XyznParser {
    static extractPositionsAndNormals(file: string): XyznPoint[] {
        const points: XyznPoint[] = [];

        for (const rawLine of file.split("\n")) {
            const line = rawLine.trim();
            if (!line || line.startsWith("#") || line.startsWith("x y z")) {
                continue;
            }

            const values = line.split(/\s+/).map((value) =>
                Number.parseFloat(value),
            );
            if (values.length !== 6 || values.some((value) => !Number.isFinite(value))) {
                throw new Error(`Invalid XYZN line: ${line}`);
            }

            points.push({
                position: vec3.create(values[0], values[1], values[2]),
                normal: vec3.normalize(
                    vec3.create(values[3], values[4], values[5]),
                ),
            });
        }

        return points;
    }
}
