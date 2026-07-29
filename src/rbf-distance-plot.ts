import Chart from "chart.js/auto";
import {
    analyzeRbfDistance,
    RbfDistanceAnalysis,
} from "./rbf-distance-analysis";
import { RbfFitConfig, RbfFitResult } from "./rbf";

interface PlotPoint {
    x: number;
    y: number;
}

export class RbfDistancePlot {
    private readonly chart: Chart<"scatter" | "line", PlotPoint[]>;
    private calculationId = 0;

    constructor(
        canvas: HTMLCanvasElement,
        private readonly statusElement: HTMLElement,
    ) {
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

                this.updateStatus(analysis);
                this.updateChart(analysis);
            }, 0);
        });
    }

    private updateChart(analysis: RbfDistanceAnalysis): void {
        const points = analysis.samples.map(({ rbf, dist }) => ({
            x: rbf,
            y: dist,
        }));
        let axisMaximum = 0;
        for (const point of points) {
            axisMaximum = Math.max(axisMaximum, point.x, point.y);
        }
        axisMaximum = axisMaximum > 0 ? axisMaximum * 1.05 : 1;

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

    private updateStatus(analysis: RbfDistanceAnalysis): void {
        if (analysis.zeroCrossingCount === 0) {
            this.statusElement.textContent =
                `${analysis.gridSampleCount.toLocaleString()} samples · no zero crossings`;
            return;
        }

        this.statusElement.textContent =
            `${analysis.gridSampleCount.toLocaleString()} samples · ` +
            `${analysis.zeroCrossingCount.toLocaleString()} zeroes · ` +
            `${analysis.elapsedMilliseconds.toFixed(1)} ms`;
    }
}
