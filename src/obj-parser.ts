interface Point {
    x: number;
    y: number;
    z: number;
}

interface OrientedPoint {
    position: Point;
    normal: Point;
}

export class ObjParser {
    static extractPositionsAndNormals(objFile: string): OrientedPoint[] {
        const lines: string[] = objFile.split("\n");
        const orientedPoints: OrientedPoint[] = [];
        let currNormal: Point = { x: 0, y: 0, z: 0 };
        for (let line of lines) {
            line = line.trim();
            const coordLineArray: string[] = line.split(/\s+/);
            const type: string = coordLineArray[0];
            if (type == "vn" && coordLineArray.length == 4) {
                const normal: Point = {
                    x: Number.parseFloat(coordLineArray[1]),
                    y: Number.parseFloat(coordLineArray[2]),
                    z: Number.parseFloat(coordLineArray[3]),
                };
                currNormal = normal;
            }
            if (type == "v" && coordLineArray.length == 4) {
                const point: Point = {
                    x: Number.parseFloat(coordLineArray[1]),
                    y: Number.parseFloat(coordLineArray[2]),
                    z: Number.parseFloat(coordLineArray[3]),
                };
                orientedPoints.push({ position: point, normal: currNormal });
            }
        }

        return orientedPoints;
    }
}
