const {
    readFileSync: readSnapshotFile,
    writeFileSync: writeComparisonFile,
} = require("node:fs");
const { basename: metricFileName } = require("node:path");

const METRIC_LAYOUT = ["distance", "stepCount", "hit"];
const STRIDE = METRIC_LAYOUT.length;
const OUTPUT_FILE_NAME = "metrics-comparisson.csv";

type Config = Record<string, unknown>;

interface Snapshot {
    width: number;
    height: number;
    sceneId: string;
    rbfConfig: Config;
    rayMarchingConfig: Config;
    metricLayout: string[];
    metrics: number[];
}

interface Stats {
    mean: number | null;
    median: number | null;
}

interface Comparison {
    candidate: string;
    configChanges: string;
    config: Config;
    pixels: number;
    agreement: number;
    falsePositives: number;
    falseNegatives: number;
    commonHits: number;
    rmse: number | null;
    baselineSteps: Stats;
    candidateSteps: Stats;
}

function loadSnapshot(path: string): Snapshot {
    const snapshot = JSON.parse(readSnapshotFile(path, "utf8")) as Snapshot;

    if (snapshot.metricLayout?.join(",") !== METRIC_LAYOUT.join(",")) {
        throw new Error(`${path}: unsupported metric layout.`);
    }

    const expectedLength = snapshot.width * snapshot.height * STRIDE;
    if (snapshot.metrics.length !== expectedLength) {
        throw new Error(
            `${path}: expected ${expectedLength} metrics, received ${snapshot.metrics.length}.`,
        );
    }

    return snapshot;
}

function getConfig(snapshot: Snapshot): Config {
    return {
        ...snapshot.rbfConfig,
        ...snapshot.rayMarchingConfig,
    };
}

function getConfigChanges(baseline: Config, candidate: Config): string {
    const changes = Object.keys(candidate)
        .filter((key) => candidate[key] !== baseline[key])
        .map((key) => `${key}: ${baseline[key]} -> ${candidate[key]}`);

    return changes.join("; ") || "None";
}

function calculateStats(values: number[]): Stats {
    if (values.length === 0) {
        return { mean: null, median: null };
    }

    values.sort((left, right) => left - right);
    const middle = Math.floor(values.length / 2);
    const median =
        values.length % 2 === 0
            ? (values[middle - 1] + values[middle]) / 2
            : values[middle];

    return {
        mean: values.reduce((sum, value) => sum + value, 0) / values.length,
        median,
    };
}

function collectBaselineSteps(snapshot: Snapshot): number[] {
    const steps: number[] = [];
    for (let index = 0; index < snapshot.metrics.length; index += STRIDE) {
        if (snapshot.metrics[index + 2] === 1) {
            steps.push(snapshot.metrics[index + 1]);
        }
    }
    return steps;
}

function compare(
    baseline: Snapshot,
    candidate: Snapshot,
    candidatePath: string,
    baselineConfig: Config,
    baselineSteps: Stats,
): Comparison {
    if (
        baseline.width !== candidate.width ||
        baseline.height !== candidate.height ||
        baseline.sceneId !== candidate.sceneId
    ) {
        throw new Error(`${candidatePath}: incompatible with baseline.`);
    }

    let agreement = 0;
    let falsePositives = 0;
    let falseNegatives = 0;
    let commonHits = 0;
    let squaredError = 0;
    const candidateStepValues: number[] = [];

    for (let index = 0; index < baseline.metrics.length; index += STRIDE) {
        const baselineHit = baseline.metrics[index + 2] === 1;
        const candidateHit = candidate.metrics[index + 2] === 1;

        if (baselineHit === candidateHit) {
            agreement++;
        } else if (candidateHit) {
            falsePositives++;
        } else {
            falseNegatives++;
        }

        if (baselineHit && candidateHit) {
            const error = candidate.metrics[index] - baseline.metrics[index];
            squaredError += error * error;
            commonHits++;
        }

        if (baselineHit) {
            candidateStepValues.push(candidate.metrics[index + 1]);
        }
    }

    const config = getConfig(candidate);
    return {
        candidate: metricFileName(candidatePath),
        configChanges: getConfigChanges(baselineConfig, config),
        config,
        pixels: baseline.width * baseline.height,
        agreement,
        falsePositives,
        falseNegatives,
        commonHits,
        rmse: commonHits === 0 ? null : Math.sqrt(squaredError / commonHits),
        baselineSteps,
        candidateSteps: calculateStats(candidateStepValues),
    };
}

function percent(value: number, total: number): number {
    return (value / total) * 100;
}

function display(value: number | null, digits: number): string {
    return value === null ? "N/A" : value.toFixed(digits);
}

function printReport(result: Comparison): void {
    console.log(`\nCandidate: ${result.candidate}`);
    console.log(`Config changes: ${result.configChanges}`);
    console.log(
        `Hit agreement: ${result.agreement}/${result.pixels} (${percent(result.agreement, result.pixels).toFixed(2)}%)`,
    );
    console.log(
        `False positives: ${result.falsePositives} (${percent(result.falsePositives, result.pixels).toFixed(2)}%)`,
    );
    console.log(
        `False negatives: ${result.falseNegatives} (${percent(result.falseNegatives, result.pixels).toFixed(2)}%)`,
    );
    console.log(
        `Distance RMSE: ${display(result.rmse, 6)} (${result.commonHits} common hits)`,
    );
    console.log(
        `Mean steps: ${display(result.baselineSteps.mean, 2)} -> ${display(result.candidateSteps.mean, 2)}`,
    );
    console.log(
        `Median steps: ${display(result.baselineSteps.median, 2)} -> ${display(result.candidateSteps.median, 2)}`,
    );
}

function printTable(results: Comparison[]): void {
    console.log("\nSummary");
    console.table(
        results.map((result) => ({
            Candidate: result.candidate,
            "Config changes": result.configChanges,
            "Hit agreement": `${percent(result.agreement, result.pixels).toFixed(2)}%`,
            FP: `${percent(result.falsePositives, result.pixels).toFixed(2)}%`,
            FN: `${percent(result.falseNegatives, result.pixels).toFixed(2)}%`,
            RMSE: display(result.rmse, 6),
            "Base mean": display(result.baselineSteps.mean, 2),
            "Candidate mean": display(result.candidateSteps.mean, 2),
            "Base median": display(result.baselineSteps.median, 2),
            "Candidate median": display(result.candidateSteps.median, 2),
        })),
    );
}

function escapeCsv(value: unknown): string {
    if (value === null || value === undefined) {
        return "";
    }
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function writeCsv(results: Comparison[]): void {
    const configKeys = [
        ...new Set(results.flatMap((result) => Object.keys(result.config))),
    ];
    const columns = [
        "candidate",
        "configChanges",
        "hitAgreementCount",
        "hitAgreementPercent",
        "falsePositiveCount",
        "falsePositivePercent",
        "falseNegativeCount",
        "falseNegativePercent",
        "commonHitCount",
        "distanceRmse",
        "baselineMeanSteps",
        "candidateMeanSteps",
        "baselineMedianSteps",
        "candidateMedianSteps",
        ...configKeys,
    ];
    const rows = results.map((result) => [
        result.candidate,
        result.configChanges,
        result.agreement,
        percent(result.agreement, result.pixels),
        result.falsePositives,
        percent(result.falsePositives, result.pixels),
        result.falseNegatives,
        percent(result.falseNegatives, result.pixels),
        result.commonHits,
        result.rmse,
        result.baselineSteps.mean,
        result.candidateSteps.mean,
        result.baselineSteps.median,
        result.candidateSteps.median,
        ...configKeys.map((key) => result.config[key]),
    ]);

    const csv = [columns, ...rows]
        .map((row) => row.map(escapeCsv).join(","))
        .join("\n");
    writeComparisonFile(OUTPUT_FILE_NAME, `${csv}\n`);
    console.log(`Created ${OUTPUT_FILE_NAME}.`);
}

function runSnapshotComparison(argv: string[]): void {
    const [baselinePath, ...candidatePaths] = argv.slice(2);
    if (!baselinePath || candidatePaths.length === 0) {
        throw new Error(
            "Usage: npm run compare-metrics -- <baseline.json> <candidate.json> [candidate.json ...]",
        );
    }

    const baseline = loadSnapshot(baselinePath);
    const baselineConfig = getConfig(baseline);
    const baselineSteps = calculateStats(collectBaselineSteps(baseline));
    const baselineResult = compare(
        baseline,
        baseline,
        baselinePath,
        baselineConfig,
        baselineSteps,
    );
    baselineResult.configChanges = "Baseline";
    baselineResult.rmse = 0;

    console.log(
        `Baseline: ${metricFileName(baselinePath)} (${baseline.width}x${baseline.height}, ${baseline.sceneId})`,
    );

    const results = [baselineResult];
    for (const path of candidatePaths) {
        const result = compare(
            baseline,
            loadSnapshot(path),
            path,
            baselineConfig,
            baselineSteps,
        );
        results.push(result);
        printReport(result);
    }

    printTable(results);
    writeCsv(results);
}

try {
    runSnapshotComparison(process.argv);
} catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
}
