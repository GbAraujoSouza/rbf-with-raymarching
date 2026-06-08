import { vec3 } from "gl-matrix";
import { createBuiltInRbfFit, type RbfFitResult } from "./rbf";
import shader from "./shaders/rbf_raymarch.wgsl";
import { DEFAULT_EXPERIMENT_STATE, ExperimentState } from "./experiment";

export class Renderer {
    canvas: HTMLCanvasElement;

    adapter!: GPUAdapter;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;

    bindGroup!: GPUBindGroup;
    pipeline!: GPURenderPipeline;

    uniformBuffer!: GPUBuffer;
    positionsBuffer!: GPUBuffer;
    weightsBuffer!: GPUBuffer;

    rbfFit!: RbfFitResult;

    experimentState: ExperimentState;

    gpuTexture!: GPUTexture;
    gpuTextureView!: GPUTextureView;

    yaw: number = 0.7;
    pitch: number = 0.5;
    radius: number = 4.2;
    readonly target: vec3 = vec3.fromValues(0, 0, 0);
    readonly fieldOfView: number = Math.PI / 4;

    isPointerDown: boolean = false;
    lastPointerX: number = 0;
    lastPointerY: number = 0;

    constructor(
        canvas: HTMLCanvasElement,
        experimentState: ExperimentState = DEFAULT_EXPERIMENT_STATE,
    ) {
        this.canvas = canvas;
        this.experimentState = experimentState;
    }

    async initialize() {
        await this.setupDevice();

        this.createAssets();

        await this.makePipeline();
        this.setupInteraction();

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
            size: 8 * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {},
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: { type: "read-only-storage" },
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
                {
                    binding: 1,
                    resource: {
                        buffer: this.positionsBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.weightsBuffer,
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
        this.rbfFit = createBuiltInRbfFit(this.experimentState.rbfConfig);
        this.positionsBuffer = this.device.createBuffer({
            size: this.rbfFit.positions.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.weightsBuffer = this.device.createBuffer({
            size: this.rbfFit.weights.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.device.queue.writeBuffer(
            this.positionsBuffer,
            0,
            this.rbfFit.positions,
        );
        this.device.queue.writeBuffer(
            this.weightsBuffer,
            0,
            this.rbfFit.weights,
        );
    }

    render = () => {
        this.updateSceneUniforms();

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
        pass.draw(3);

        pass.end();
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        requestAnimationFrame(this.render);
    };

    updateSceneUniforms() {
        const eye = vec3.fromValues(
            this.radius * Math.cos(this.pitch) * Math.sin(this.yaw),
            this.radius * Math.sin(this.pitch),
            this.radius * Math.cos(this.pitch) * Math.cos(this.yaw),
        );
        const forward = vec3.create();
        vec3.subtract(forward, this.target, eye);
        vec3.normalize(forward, forward);

        const worldUp = vec3.fromValues(0, 1, 0);
        const right = vec3.create();
        vec3.cross(right, forward, worldUp);
        vec3.normalize(right, right);

        const up = vec3.create();
        vec3.cross(up, right, forward);
        vec3.normalize(up, up);

        const uniformData = new Float32Array(8 * 4);
        uniformData.set(
            [
                this.canvas.width,
                this.canvas.height,
                this.rbfFit.weights.length,
                this.experimentState.showControlPoints ? 1 : 0,
            ],
            0,
        );
        uniformData.set([eye[0], eye[1], eye[2], 0], 4);
        uniformData.set([forward[0], forward[1], forward[2], 0], 8);
        uniformData.set([right[0], right[1], right[2], 0], 12);
        uniformData.set([up[0], up[1], up[2], 0], 16);
        uniformData.set(
            [
                this.experimentState.rayMarchingConfig.correctionPower,
                this.experimentState.rayMarchingConfig.correctionLinear,
                this.experimentState.rayMarchingConfig.maxDistance,
                this.experimentState.rayMarchingConfig.epsilon,
            ],
            20,
        );

        // rbf params
        uniformData.set(
            [
                this.experimentState.rbfConfig.gaussianEpsilon,
                this.experimentState.rbfConfig.debugPointRadius,
                this.fieldOfView,
                0,
            ],
            24,
        );
        uniformData.set([10, 10, 10, 0], 28);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    }

    setupInteraction() {
        this.canvas.addEventListener("pointerdown", (event) => {
            this.isPointerDown = true;
            this.lastPointerX = event.clientX;
            this.lastPointerY = event.clientY;
            this.canvas.setPointerCapture(event.pointerId);
        });

        this.canvas.addEventListener("pointermove", (event) => {
            if (!this.isPointerDown) {
                return;
            }

            const deltaX = event.clientX - this.lastPointerX;
            const deltaY = event.clientY - this.lastPointerY;
            const rotationSpeed = 0.01;

            this.yaw -= deltaX * rotationSpeed;
            this.pitch = clamp(
                this.pitch - deltaY * rotationSpeed,
                -1.45,
                1.45,
            );

            this.lastPointerX = event.clientX;
            this.lastPointerY = event.clientY;
        });

        const releasePointer = (event: PointerEvent) => {
            this.isPointerDown = false;
            if (this.canvas.hasPointerCapture(event.pointerId)) {
                this.canvas.releasePointerCapture(event.pointerId);
            }
        };

        this.canvas.addEventListener("pointerup", releasePointer);
        this.canvas.addEventListener("pointerleave", releasePointer);
        this.canvas.addEventListener("pointercancel", releasePointer);
    }
}

function clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), max);
}
