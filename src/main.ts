import "./styles.css";
import { Renderer } from "./renderer";
import { setupUI } from "./interface";
import { ObjParser } from "./obj-parser";

const canvas = <HTMLCanvasElement>(
    document.querySelector<HTMLCanvasElement>("#gpu-canvas")
);

if (!canvas) {
    throw new Error("Missing required DOM nodes.");
}

canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const renderer = new Renderer(canvas);

renderer.initialize();
setupUI(renderer);
