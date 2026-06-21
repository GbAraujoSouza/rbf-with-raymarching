interface Point {
    x: number;
    y: number;
    z: number;
}

export class ObjParser {
    static extractPositions(objFile: string): Point[] {
        const lines: string[] = objFile.split("\n");
        const positions: Point[] = [];
        for (let line of lines) {
            line = line.trim();
            const coordLineArray: string[] = line.split(/\s+/);
            const type: string = coordLineArray[0];
            if (type == "v" && coordLineArray.length == 4) {
                const point: Point = {
                    x: Number.parseFloat(coordLineArray[1]),
                    y: Number.parseFloat(coordLineArray[2]),
                    z: Number.parseFloat(coordLineArray[3]),
                };
                positions.push(point);
            }
        }

        return positions;
    }
}
