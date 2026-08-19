const { readFileSync, writeFileSync } = require("node:fs");

const WIDTH = 1920;
const HEIGHT = 1080;
const MAX_STEPS = 32;
const INPUT_FILE = "metrics.txt";
const OUTPUT_FILE = "heatmap.ppm";

function readMetrics(inputFile: String): number[] {
    return readFileSync(inputFile, "utf8").split(",").map(Number);
}

function createHeatmap(metrics: number[]): Buffer {
    const pixels = Buffer.alloc(metrics.length * 3);

    for (let index = 0; index < metrics.length; index++) {
        const intensity = Math.min(metrics[index] / MAX_STEPS, 1);
        const pixelOffset = index * 3;

        pixels[pixelOffset] = Math.round(intensity * 255);
        pixels[pixelOffset + 1] = 0;
        pixels[pixelOffset + 2] = Math.round((1 - intensity) * 255);
    }

    return pixels;
}

function main(argv: String[]): void {
    const inputFile = argv[2] ?? INPUT_FILE;
    const compareFile = argv[3] ?? INPUT_FILE;

    const metrics = readMetrics(inputFile);
    const metrics2 = readMetrics(compareFile);
    const expectedMetrics = WIDTH * HEIGHT;

    if (metrics.length !== expectedMetrics) {
        throw new Error(
            `Expected ${expectedMetrics} metrics, received ${metrics.length}.`,
        );
    }

    const differences = metrics.map((value, index) => {
        return Math.abs(value - metrics2[index]);
    });

    const header = Buffer.from(`P6\n${WIDTH} ${HEIGHT}\n255\n`);
    const heatmap = Buffer.concat([header, createHeatmap(differences)]);

    writeFileSync(OUTPUT_FILE, heatmap);
    console.log(`Created ${OUTPUT_FILE} (${WIDTH}x${HEIGHT}).`);
}

main(process.argv);
