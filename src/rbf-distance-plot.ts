import Chart from "chart.js/auto";
import {
    analyzeRbfDistance,
    RbfDistanceAnalysis,
} from "./rbf-distance-analysis";
import { RbfFitConfig, RbfFitResult } from "./rbf";
import { RayMarchingConfig, StepStrategy } from "./experiment";

interface PlotPoint {
    x: number;
    y: number;
}

export class RbfDistancePlot {
    private readonly chart: Chart<"scatter" | "line", PlotPoint[]>;
    private calculationId = 0;
    private analysis?: RbfDistanceAnalysis;
    private rayMarchingConfig: RayMarchingConfig;

    constructor(
        canvas: HTMLCanvasElement,
        private readonly statusElement: HTMLElement,
        rayMarchingConfig: RayMarchingConfig,
    ) {
        this.rayMarchingConfig = { ...rayMarchingConfig };
        this.chart = new Chart<"scatter" | "line", PlotPoint[]>(canvas, {
            type: "scatter",
            data: {
                datasets: [
                    {
                        label: "Samples",
                        data: [],
                        pointBackgroundColor: "rgba(255, 255, 255, 0.42)",
                        pointBorderWidth: 0,
                        pointRadius: 1,
                    },
                    {
                        type: "line",
                        label: "Ideal distance",
                        data: [],
                        borderColor: "#fb7185",
                        borderWidth: 1.5,
                        pointRadius: 0,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                parsing: false,
                normalized: true,
                events: [],
                plugins: {
                    legend: {
                        labels: {
                            color: "rgba(255, 255, 255, 0.75)",
                        },
                    },
                },
                scales: {
                    x: {
                        type: "linear",
                        min: 0,
                        title: {
                            display: true,
                            text: "|RBF|",
                            color: "rgba(255, 255, 255, 0.8)",
                        },
                        ticks: {
                            color: "rgba(255, 255, 255, 0.65)",
                        },
                        grid: {
                            color: "rgba(255, 255, 255, 0.12)",
                        },
                    },
                    y: {
                        type: "linear",
                        min: 0,
                        title: {
                            display: true,
                            text: "Distance",
                            color: "rgba(255, 255, 255, 0.8)",
                        },
                        ticks: {
                            color: "rgba(255, 255, 255, 0.65)",
                        },
                        grid: {
                            color: "rgba(255, 255, 255, 0.12)",
                        },
                    },
                },
            },
        });
    }

    scheduleAnalysis(fit: RbfFitResult, config: RbfFitConfig): void {
        const calculationId = ++this.calculationId;
        const configSnapshot = { ...config };
        this.statusElement.textContent = "Calculating...";

        requestAnimationFrame(() => {
            window.setTimeout(() => {
                if (calculationId !== this.calculationId) {
                    return;
                }

                const analysis = analyzeRbfDistance(fit, configSnapshot);
                if (calculationId !== this.calculationId) {
                    return;
                }

                this.analysis = analysis;
                this.updateStatus(analysis);
                this.updateChart(analysis);
            }, 0);
        });
    }

    updateStepStrategy(config: RayMarchingConfig): void {
        this.rayMarchingConfig = { ...config };
        if (this.analysis) {
            this.updateChart(this.analysis);
        }
    }

    private updateChart(analysis: RbfDistanceAnalysis): void {
        const points = analysis.samples.map((sample) => ({
            x: this.calculateStep(sample),
            y: sample.dist,
        }));
        let axisMaximum = 2;
        // for (const point of points) {
        //     axisMaximum = Math.max(axisMaximum, point.x, point.y);
        // }
        // axisMaximum = axisMaximum > 0 ? axisMaximum * 1.05 : 1;

        this.chart.data.datasets[0].label = this.stepDatasetLabel();
        this.chart.data.datasets[0].data = points;
        this.chart.data.datasets[1].data =
            points.length > 0
                ? [
                      { x: 0, y: 0 },
                      { x: axisMaximum, y: axisMaximum },
                  ]
                : [];
        this.chart.options.scales!.x!.max = axisMaximum;
        this.chart.options.scales!.y!.max = axisMaximum;
        this.chart.update("none");
    }

    private calculateStep(
        sample: RbfDistanceAnalysis["samples"][number],
    ): number {
        const rbf = sample.rbf;
        if (
            this.rayMarchingConfig.strategy ===
            StepStrategy.exponentialCorrection
        ) {
            const correctedStep =
                this.rayMarchingConfig.correctionLinear *
                Math.pow(
                    Math.max(rbf, this.rayMarchingConfig.epsilon),
                    this.rayMarchingConfig.correctionPower,
                );
            return Math.max(correctedStep, this.rayMarchingConfig.epsilon);
        }

        if (this.rayMarchingConfig.strategy === StepStrategy.gradient) {
            const rawStep = rbf / Math.max(sample.gradientMagnitude, 0.01);
            return Math.max(rawStep, this.rayMarchingConfig.epsilon);
        }

        return Math.max(rbf, this.rayMarchingConfig.epsilon);
    }

    private stepDatasetLabel(): string {
        switch (this.rayMarchingConfig.strategy) {
            case StepStrategy.naive:
                return "Naive step";
            case StepStrategy.exponentialCorrection:
                return "Exponential step";
            case StepStrategy.gradient:
                return "Gradient step";
            case StepStrategy.cellLocalLipschitz:
                return "Raw |RBF| (Lipschitz not modeled)";
        }
    }

    private updateStatus(analysis: RbfDistanceAnalysis): void {
        if (analysis.zeroCrossingCount === 0) {
            this.statusElement.textContent = `${analysis.gridSampleCount.toLocaleString()} samples · no zero crossings`;
            return;
        }

        this.statusElement.textContent =
            `${analysis.gridSampleCount.toLocaleString()} samples · ` +
            // `${analysis.zeroCrossingCount.toLocaleString()} zeroes · ` +
            `${analysis.elapsedMilliseconds.toFixed(1)} ms`;
    }
}
