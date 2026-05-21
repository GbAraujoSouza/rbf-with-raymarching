import { glMatrix, mat4 } from "gl-matrix";
import shader from "./shaders/cell.wgsl";
import { TriangularMesh } from "./triangular_mesh";

export class Renderer {
    canvas: HTMLCanvasElement;

    adapter!: GPUAdapter;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;

    bindGroup!: GPUBindGroup;
    pipeline!: GPURenderPipeline;

    uniformBuffer!: GPUBuffer;
    t: number = 0;

    // assets
    triangularMesh!: TriangularMesh;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async initialize() {
        await this.setupDevice();

        this.createAssets();

        await this.makePipeline();

        const observer = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const canvas: HTMLCanvasElement =
                    entry.target as HTMLCanvasElement;
                const width = entry.contentBoxSize[0].inlineSize;
                const height = entry.contentBoxSize[0].blockSize;
                canvas.width = Math.max(
                    1,
                    Math.min(width, this.device.limits.maxTextureDimension2D),
                );
                canvas.height = Math.max(
                    1,
                    Math.min(height, this.device.limits.maxTextureDimension2D),
                );
            }
        });
        observer.observe(this.canvas);

        requestAnimationFrame(this.render);
    }

    async setupDevice() {
        if (!("gpu" in navigator)) {
            console.log(
                "WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari Technology Preview.",
            );
            return;
        }

        this.adapter = <GPUAdapter>await navigator.gpu.requestAdapter();

        if (!this.adapter) {
            console.log(
                "The browser supports WebGPU, but no GPU adapter was returned.",
            );
            return;
        }

        this.device = await this.adapter.requestDevice();

        this.context = <GPUCanvasContext>this.canvas.getContext("webgpu");

        if (!this.context) {
            console.log("Could not create a WebGPU canvas context.");
            return;
        }

        //const format = navigator.gpu.getPreferredCanvasFormat();
        this.format = "bgra8unorm";

        this.context.configure({
            alphaMode: "opaque",
            device: this.device,
            format: this.format,
        });
    }

    async makePipeline() {
        this.uniformBuffer = this.device.createBuffer({
            size: 3 * 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {},
                },
            ],
        });

        this.bindGroup = this.device.createBindGroup({
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
            ],
            layout: bindGroupLayout,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        const shaderModule = this.device.createShaderModule({
            code: shader,
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,

            vertex: {
                module: shaderModule,
                entryPoint: "vertexMain",
                buffers: [this.triangularMesh.bufferLayout],
            },

            fragment: {
                module: shaderModule,
                entryPoint: "fragmentMain",
                targets: [{ format: this.format }],
            },

            primitive: {
                topology: "triangle-list",
            },
        });
    }

    createAssets() {
        this.triangularMesh = new TriangularMesh(this.device);
    }

    render = () => {
        this.t += 1;

        const projection = mat4.create();
        mat4.perspectiveZO(
            projection,
            glMatrix.toRadian(45),
            800 / 600,
            0.1,
            100,
        );

        const view = mat4.create();
        mat4.lookAt(view, [0, 2, -2], [0, 0, 0], [0, 1, 0]);

        const model = mat4.create();
        mat4.rotate(model, model, glMatrix.toRadian(this.t), [0, 1, 0]);

        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            <ArrayBuffer>(<unknown>model),
        );
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            64 * 1,
            <ArrayBuffer>(<unknown>view),
        );
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            64 * 2,
            <ArrayBuffer>(<unknown>projection),
        );

        const encoder = this.device.createCommandEncoder();

        // for every step start a render pass
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.context.getCurrentTexture().createView(),
                    loadOp: "clear",
                    clearValue: [0.1176, 0.1176, 0.1804, 1.0],
                    storeOp: "store",
                },
            ],
        });

        // draw stuff
        pass.setPipeline(this.pipeline);

        pass.setBindGroup(0, this.bindGroup);

        pass.setVertexBuffer(0, this.triangularMesh.buffer);

        pass.draw(this.triangularMesh.vertices.length / 6);

        pass.end();
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        requestAnimationFrame(this.render);
    };
}
