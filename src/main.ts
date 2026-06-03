import "./styles.css";
import { Renderer } from "./renderer";

const canvas = <HTMLCanvasElement>(
    document.querySelector<HTMLCanvasElement>("#gpu-canvas")
);

if (!canvas) {
    throw new Error("Missing required DOM nodes.");
}

const renderer = new Renderer(canvas);

renderer.initialize();
