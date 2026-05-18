import "./styles.css";
import cellShaderCode from "./shaders/cell.wgsl";
import { TriangularMesh } from "./triangular_mesh";

const canvas = document.querySelector<HTMLCanvasElement>(
    "#gpu-canvas",
) as HTMLCanvasElement;

if (!canvas) {
    throw new Error("Missing required DOM nodes.");
}

const gpuCanvas = canvas;

async function initWebGpu() {
    if (!("gpu" in navigator)) {
        console.log(
            "WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari Technology Preview.",
        );
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        console.log(
            "The browser supports WebGPU, but no GPU adapter was returned.",
        );
        return;
    }

    const device = await adapter.requestDevice();
    const context = gpuCanvas.getContext("webgpu") as GPUCanvasContext | null;

    if (!context) {
        console.log("Could not create a WebGPU canvas context.");
        return;
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    const configure = () => {
        context.configure({
            alphaMode: "opaque",
            device,
            format: presentationFormat,
        });
    };

    configure();
    window.addEventListener("resize", configure);

    // Define constants to render game of life
    const GRID_SIZE = 16;
    const UPDATE_INTERVAL = 300; // in ms
    let step = 0; // simulation step counter
    const WORKGROUP_SIZE = 8;

    const triangularMesh: TriangularMesh = new TriangularMesh(device);

    // create the shader module with VERTEX and FRAGMENT shader
    // The code is imported from shaders/cell.wgsl
    const cellShaderModule = device.createShaderModule({
        label: "cell_shader",
        code: cellShaderCode,
    });

    const bindGroupLayout = device.createBindGroupLayout({
        label: "cell_bind_group_layout",
        entries: [],
    });

    const pipelineLayout = device.createPipelineLayout({
        label: "cell_pipeline_layout",
        bindGroupLayouts: [bindGroupLayout],
    });

    const cellPipeline = device.createRenderPipeline({
        label: "cell_pipeline",
        layout: pipelineLayout,
        vertex: {
            module: cellShaderModule,
            entryPoint: "vertexMain",
            buffers: [triangularMesh.bufferLayout],
        },
        fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{ format: presentationFormat }],
        },
    });

    const render = () => {
        const encoder = device.createCommandEncoder();

        // for every step start a render pass
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    clearValue: [0.1176, 0.1176, 0.1804, 1.0],
                    storeOp: "store",
                },
            ],
        });

        // draw stuff
        pass.setPipeline(cellPipeline);

        pass.setVertexBuffer(0, triangularMesh.buffer);

        pass.draw(triangularMesh.vertices.length / 6);

        pass.end();
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
    };

    setInterval(render, UPDATE_INTERVAL);
    console.log("WebGPU initialized successfully.");

    const observer = new ResizeObserver((entries) => {
        for (const entry of entries) {
            const canvas: HTMLCanvasElement = entry.target as HTMLCanvasElement;
            const width = entry.contentBoxSize[0].inlineSize;
            const height = entry.contentBoxSize[0].blockSize;
            canvas.width = Math.max(
                1,
                Math.min(width, device.limits.maxTextureDimension2D),
            );
            canvas.height = Math.max(
                1,
                Math.min(height, device.limits.maxTextureDimension2D),
            );
        }
        // re-render
        render();
    });
    observer.observe(canvas);
}

initWebGpu().catch((error: unknown) => {
    console.error(error);
    const message = error instanceof Error ? error.message : "Unknown error";
    console.error(`Failed to initialize WebGPU: ${message}`);
});
