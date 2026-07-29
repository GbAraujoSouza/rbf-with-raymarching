import "./styles.css";
import { Renderer } from "./renderer";
import { setupUI } from "./interface";
import { RbfDistancePlot } from "./rbf-distance-plot";

const canvas = <HTMLCanvasElement>(
    document.querySelector<HTMLCanvasElement>("#gpu-canvas")
);

if (!canvas) {
    throw new Error("Missing required DOM nodes.");
}

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const renderer = new Renderer(canvas);

async function initializeApplication(): Promise<void> {
    await renderer.initialize();
    setupUI(renderer);

    const plotCanvas = document.querySelector<HTMLCanvasElement>(
        "#rbf-distance-plot",
    );
    const plotStatus = document.querySelector<HTMLElement>(
        "#rbf-distance-status",
    );
    if (!plotCanvas || !plotStatus) {
        throw new Error("Missing required RBF distance plot nodes.");
    }

    const distancePlot = new RbfDistancePlot(plotCanvas, plotStatus);
    renderer.onRbfFitChanged((fit, config) => {
        distancePlot.scheduleAnalysis(fit, config);
    });
}

void initializeApplication();
