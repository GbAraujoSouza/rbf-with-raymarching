import "./styles.css";
import { Renderer } from "./renderer";
import { Scene } from "./scene";

const canvas = <HTMLCanvasElement>(
    document.querySelector<HTMLCanvasElement>("#gpu-canvas")
);

if (!canvas) {
    throw new Error("Missing required DOM nodes.");
}

// Define a resolução interna do canvas baseada na janela atual,
// substituindo o tamanho padrão minúsculo (300x150) do navegador.
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const scene: Scene = new Scene(1024 * 4);

const renderer = new Renderer(canvas, scene);

renderer.initialize();
