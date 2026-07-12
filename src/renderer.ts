import { mat4, Mat4, vec3, Vec3 } from "wgpu-matrix";
import { createBuiltInRbfFit, type RbfFitResult } from "./rbf";
import screenShaderCode from "./shaders/screen_shader.wgsl";
import computeShaderCode from "./shaders/ray_marching_compute.wgsl";
import { DEFAULT_EXPERIMENT_STATE, ExperimentState } from "./experiment";
import {
    SCENE_UNIFORM_BYTES,
    SceneUniforms,
    SceneUniformInput,
} from "./scene-uniforms";

export class Renderer {
    canvas: HTMLCanvasElement;

    adapter!: GPUAdapter;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;

    bindGroup!: GPUBindGroup;
    pipeline!: GPURenderPipeline;

    computeBindGroup!: GPUBindGroup;
    computePipeline!: GPUComputePipeline;

    uniformBuffer!: GPUBuffer;
    positionsBuffer!: GPUBuffer;
    weightsBuffer!: GPUBuffer;

    rbfFit!: RbfFitResult;

    experimentState: ExperimentState;

    gpuTexture!: GPUTexture;
    gpuTextureView!: GPUTextureView;
    sampler!: GPUSampler;

    objectToWorld!: Mat4;
    worldToObject!: Mat4;

    yaw: number = 0.7;
    pitch: number = 0.5;
    radius: number = 10;
    readonly target: Vec3 = vec3.fromValues(0, 0, 0);
    readonly fieldOfView: number = Math.PI / 4;

    isPointerDown: boolean = false;
    lastPointerX: number = 0;
    lastPointerY: number = 0;

    lastTime: number = performance.now();
    frameCount: number = 0;
    fps: number = 0;
    fpsElement: HTMLElement | null = null;

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

        const context = this.canvas.getContext(
            "webgpu",
        ) as GPUCanvasContext | null;

        if (!context) {
            console.log("Could not create a WebGPU canvas context.");
            return;
        }
        this.context = context;

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
            size: SCENE_UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const computeBindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        access: "write-only",
                        format: "rgba8unorm",
                        viewDimension: "2d",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
            ],
        });

        this.computeBindGroup = this.device.createBindGroup({
            layout: computeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.gpuTextureView,
                },
                {
                    binding: 1,
                    resource: this.uniformBuffer,
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.positionsBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.weightsBuffer,
                    },
                },
            ],
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    sampler: {},
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {},
                },
            ],
        });

        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.sampler,
                },
                {
                    binding: 1,
                    resource: this.gpuTextureView,
                },
            ],
        });

        // CREATE PIPELINES

        const computePipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [computeBindGroupLayout],
        });

        const computeModule = this.device.createShaderModule({
            code: computeShaderCode,
        });

        this.computePipeline = this.device.createComputePipeline({
            layout: computePipelineLayout,
            compute: {
                module: computeModule,
                entryPoint: "main",
            },
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        const shaderModule = this.device.createShaderModule({
            code: screenShaderCode,
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
        this.rbfFit = createBuiltInRbfFit(
            this.experimentState.rbfConfig,
            this.experimentState.sceneId,
        );
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

        this.gpuTexture = this.device.createTexture({
            size: {
                width: this.canvas.width,
                height: this.canvas.height,
            },
            format: "rgba8unorm",
            usage:
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING,
        });

        this.gpuTextureView = this.gpuTexture.createView();

        const samplerDescriptor: GPUSamplerDescriptor = {
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "nearest",
            mipmapFilter: "nearest",
            maxAnisotropy: 1,
        };

        this.sampler = this.device.createSampler(samplerDescriptor);
    }

    render = () => {
        const now = performance.now();
        this.frameCount++;
        if (now - this.lastTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastTime = now;
            if (!this.fpsElement) {
                this.fpsElement = document.getElementById("fps-counter");
            }
            if (this.fpsElement) {
                this.fpsElement.innerText = this.fps.toString();
            }
        }

        this.updateSceneUniforms();

        const encoder = this.device.createCommandEncoder();

        const computePass: GPUComputePassEncoder = encoder.beginComputePass();
        computePass.setBindGroup(0, this.computeBindGroup);
        computePass.setPipeline(this.computePipeline);
        computePass.dispatchWorkgroups(
            Math.ceil(this.canvas.width / 16),
            Math.ceil(this.canvas.height / 16),
            1,
        );
        computePass.end();

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
        pass.draw(6, 1, 0, 0);

        pass.end();
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        requestAnimationFrame(this.render);
    };

    buildTransformations() {
        this.objectToWorld = mat4.translation(vec3.create(0.0, 0.0, 0.0));
        this.worldToObject = mat4.inverse(this.objectToWorld);
    }

    updateSceneUniforms() {
        this.buildTransformations();

        const eye: Vec3 = vec3.fromValues(
            this.radius * Math.cos(this.pitch) * Math.sin(this.yaw),
            this.radius * Math.sin(this.pitch),
            this.radius * Math.cos(this.pitch) * Math.cos(this.yaw),
        );
        const forward: Vec3 = vec3.create();
        vec3.subtract(this.target, eye, forward);
        vec3.normalize(forward, forward);

        const worldUp: Vec3 = vec3.fromValues(0, 1, 0);
        const right: Vec3 = vec3.create();
        vec3.cross(forward, worldUp, right);
        vec3.normalize(right, right);

        const up: Vec3 = vec3.create();
        vec3.cross(right, forward, up);
        vec3.normalize(up, up);

        const uniformInput: SceneUniformInput = {
            screenWidth: this.canvas.width,
            screenHeight: this.canvas.height,

            sampleCount: this.rbfFit.samples.length,
            debugPoints: this.experimentState.showControlPoints ? 1 : 0,

            cameraPosition: eye,
            cameraForward: forward,
            cameraRight: right,
            cameraUp: up,

            gaussianKernelCorrectionPower:
                this.experimentState.rayMarchingConfig.correctionPower,
            gaussianKernelCorrectionLinear:
                this.experimentState.rayMarchingConfig.correctionLinear,
            maxDistance: this.experimentState.rayMarchingConfig.maxDistance,
            epsilon: this.experimentState.rayMarchingConfig.epsilon,

            gaussianEpsilon: this.experimentState.rbfConfig.gaussianEpsilon,
            debugPointRadius: this.experimentState.rbfConfig.debugPointRadius,
            stepStrategy: this.experimentState.rayMarchingConfig.strategy,
            renderMode: this.experimentState.renderMode,

            lightPosition: vec3.create(10, 10, 10),
            kernelType: this.experimentState.rbfConfig.kernel,

            boxMin: this.rbfFit.boxMin,
            boxMax: this.rbfFit.boxMax,
        };

        const uniformData: Float32Array<ArrayBuffer> =
            SceneUniforms.createSceneUniformData(uniformInput);

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

    rebuildRbfAssets() {
        this.rbfFit = createBuiltInRbfFit(
            this.experimentState.rbfConfig,
            this.experimentState.sceneId,
        );

        this.positionsBuffer.destroy();
        this.weightsBuffer.destroy();

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

        this.computeBindGroup = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.gpuTextureView,
                },
                {
                    binding: 1,
                    resource: { buffer: this.uniformBuffer },
                },
                {
                    binding: 2,
                    resource: { buffer: this.positionsBuffer },
                },
                {
                    binding: 3,
                    resource: { buffer: this.weightsBuffer },
                },
            ],
        });
    }
}

function clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), max);
}
