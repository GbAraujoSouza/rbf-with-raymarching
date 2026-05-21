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

// const observer = new ResizeObserver((entries) => {
//     for (const entry of entries) {
//         const canvas: HTMLCanvasElement = entry.target as HTMLCanvasElement;
//         const width = entry.contentBoxSize[0].inlineSize;
//         const height = entry.contentBoxSize[0].blockSize;
//         canvas.width = Math.max(
//             1,
//             Math.min(width, device.limits.maxTextureDimension2D),
//         );
//         canvas.height = Math.max(
//             1,
//             Math.min(height, device.limits.maxTextureDimension2D),
//         );
//     }
//     // re-render
//     render();
// });
// observer.observe(canvas);
