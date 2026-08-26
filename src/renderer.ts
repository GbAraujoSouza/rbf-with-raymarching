import { mat4, Mat4, vec3, Vec3 } from "wgpu-matrix";
import { createBuiltInRbfFit, RbfFitConfig, type RbfFitResult } from "./rbf";
import screenShaderCode from "./shaders/screen_shader.wgsl";
import computeShaderCode from "./shaders/ray_marching_compute.wgsl";
import {
    DEFAULT_EXPERIMENT_STATE,
    ExperimentState,
    RenderBackend,
    StepStrategy,
} from "./experiment";
import {
    SCENE_UNIFORM_BYTES,
    SceneUniforms,
    SceneUniformInput,
} from "./scene-uniforms";
import { MeshCameraUniformInput, MeshRenderer } from "./mesh-renderer";
import { buildMarchingCubesMesh, ExtractedMesh } from "./mesh";
import {
    buildLipschitzGrid,
    LipschitzGrid,
    LipschitzGridConfig,
} from "./lipschitz-grid";

export class Renderer {
    canvas: HTMLCanvasElement;

    adapter!: GPUAdapter;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;

    bindGroup!: GPUBindGroup;
    pipeline!: GPURenderPipeline;

    computeBindGroup!: GPUBindGroup;
    computeBindGroupLayout!: GPUBindGroupLayout;
    computePipeline!: GPUComputePipeline;

    uniformBuffer!: GPUBuffer;
    positionsBuffer!: GPUBuffer;
    weightsBuffer!: GPUBuffer;
    lipschitzBuffer!: GPUBuffer;
    metricsWorkBuffer!: GPUBuffer;
    metricsResultBuffer!: GPUBuffer;

    rbfFit!: RbfFitResult;
    lipschitzGrid?: LipschitzGrid;

    experimentState: ExperimentState;

    gpuTexture!: GPUTexture;
    gpuTextureView!: GPUTextureView;
    sampler!: GPUSampler;

    objectToWorld!: Mat4;
    worldToObject!: Mat4;

    meshRenderer!: MeshRenderer;

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
    private readonly rbfFitListeners = new Set<
        (fit: RbfFitResult, config: RbfFitConfig) => void
    >();

    private copyMetricsToBuffer: boolean = false;

    constructor(
        canvas: HTMLCanvasElement,
        experimentState: ExperimentState = DEFAULT_EXPERIMENT_STATE,
    ) {
        this.canvas = canvas;
        this.experimentState = experimentState;
    }

    rebuildMarchingCubesMesh(): void {
        const fit: RbfFitResult = this.rbfFit;
        const config: RbfFitConfig = this.experimentState.rbfConfig;

        const mesh: ExtractedMesh = buildMarchingCubesMesh(
            fit,
            config,
            this.experimentState.marchingCubesConfig.resolution,
            this.experimentState.marchingCubesConfig.isoValue,
            this.experimentState.marchingCubesConfig.extraPadding,
        );

        this.meshRenderer.setMesh(mesh);
    }

    captureMetrics(): void {
        if (
            this.experimentState.renderBackend !== RenderBackend.rayMarching
        ) {
            console.warn("Metrics capture is only available for Ray Marching.");
            return;
        }

        this.copyMetricsToBuffer = true;
    }

    async initialize() {
        await this.setupDevice();

        this.meshRenderer = new MeshRenderer(this.device, this.format);
        this.meshRenderer.initialize();

        this.createAssets();

        // same as createAssets for marching cubes
        this.rebuildMarchingCubesMesh();

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

        this.computeBindGroupLayout = this.device.createBindGroupLayout({
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
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
            ],
        });

        this.rebuildComputeBindGroup();

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
            bindGroupLayouts: [this.computeBindGroupLayout],
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

        const metricsBufferSize =
            this.canvas.width *
            this.canvas.height *
            2 *
            Float32Array.BYTES_PER_ELEMENT;

        // Two f32 values per pixel: traveled distance and step count.
        this.metricsWorkBuffer = this.device.createBuffer({
            size: metricsBufferSize,
            usage:
                GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_DST |
                GPUBufferUsage.COPY_SRC,
        });

        this.metricsResultBuffer = this.device.createBuffer({
            size: metricsBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Not yet initialized buffer, so start with 1 value
        this.lipschitzBuffer = this.createLipschitzBuffer(
            new Float32Array([1]),
        );

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

        if (
            this.experimentState.rayMarchingConfig.strategy ===
            StepStrategy.cellLocalLipschitz
        ) {
            this.rebuildLipschitzGrid(false);
        }

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

    render = async () => {
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

        if (this.experimentState.renderBackend == RenderBackend.rayMarching) {
            const computePass: GPUComputePassEncoder =
                encoder.beginComputePass();
            computePass.setBindGroup(0, this.computeBindGroup);
            computePass.setPipeline(this.computePipeline);
            computePass.dispatchWorkgroups(
                Math.ceil(this.canvas.width / 16),
                Math.ceil(this.canvas.height / 16),
                1,
            );
            computePass.end();

            if (this.shouldCaptureMetrics()) {
                encoder.copyBufferToBuffer(
                    this.metricsWorkBuffer,
                    this.metricsResultBuffer,
                    this.metricsResultBuffer.size,
                );
            }

            // for every step start a render pass
            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: this.context.getCurrentTexture().createView(),
                        loadOp: "clear",
                        clearValue: [0.8118, 0.9333, 1.0, 1.0],
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

            if (this.shouldCaptureMetrics()) {
                await this.unmapToBuffer();
            }
        }

        if (this.experimentState.renderBackend == RenderBackend.marchingCubes) {
            this.meshRenderer.resize(this.canvas.width, this.canvas.height);

            const pass = encoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: this.context.getCurrentTexture().createView(),
                        loadOp: "clear",
                        clearValue: [0.8118, 0.9333, 1.0, 1.0],
                        storeOp: "store",
                    },
                ],
                depthStencilAttachment: {
                    view: this.meshRenderer.depthTextureView,
                    depthClearValue: 1.0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });

            const uniforms = this.createMeshCameraUniformInput();
            this.meshRenderer.render(pass, uniforms);

            const commandBuffer = encoder.finish();
            this.device.queue.submit([commandBuffer]);
        }

        requestAnimationFrame(this.render);
    };

    private async unmapToBuffer() {
        await this.metricsResultBuffer.mapAsync(GPUMapMode.READ);
        const result = new Float32Array(
            this.metricsResultBuffer.getMappedRange().slice(),
        );
        this.metricsResultBuffer.unmap();

        const snapshot = {
            version: 1,
            width: this.canvas.width,
            height: this.canvas.height,
            sceneId: this.experimentState.sceneId,
            rbfConfig: this.experimentState.rbfConfig,
            rayMarchingConfig: this.experimentState.rayMarchingConfig,
            metricLayout: ["distance", "stepCount"],
            metrics: Array.from(result),
        };

        await fetch("/write-metrics", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(snapshot),
        });

        this.copyMetricsToBuffer = false;
    }

    private shouldCaptureMetrics(): boolean {
        return (
            this.experimentState.renderBackend == RenderBackend.rayMarching &&
            this.copyMetricsToBuffer
        );
    }

    /**
     * Setup camera position (eye position)
     * and model - view - projection matrices.
     * Also creates a fixed light
     * @returns MeshCameraUniformInput
     */
    createMeshCameraUniformInput(): MeshCameraUniformInput {
        const eye: Vec3 = vec3.fromValues(
            this.radius * Math.cos(this.pitch) * Math.sin(this.yaw),
            this.radius * Math.sin(this.pitch),
            this.radius * Math.cos(this.pitch) * Math.cos(this.yaw),
        );

        const worldUp: Vec3 = vec3.fromValues(0, 1, 0);
        const view: Mat4 = mat4.lookAt(eye, this.target, worldUp);
        const aspect = this.canvas.width / this.canvas.height;
        const projection: Mat4 = mat4.perspective(
            this.fieldOfView,
            aspect,
            0.1,
            this.experimentState.rayMarchingConfig.maxDistance,
        );

        return {
            viewProjection: mat4.multiply(projection, view),
            model: this.objectToWorld,
            cameraPosition: eye,
            lightPosition: vec3.create(10, 10, 10),
        };
    }

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

            worldToObject: this.worldToObject,

            maxSteps: this.experimentState.rayMarchingConfig.maxSteps,
            fieldOfView: this.fieldOfView,
            lipschitzGridDimensions: this.lipschitzGrid
                ? vec3.create(
                      this.lipschitzGrid.nx,
                      this.lipschitzGrid.ny,
                      this.lipschitzGrid.nz,
                  )
                : vec3.create(1, 1, 1),
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

        this.invalidateLipschitzGrid();
        if (
            this.experimentState.rayMarchingConfig.strategy ===
            StepStrategy.cellLocalLipschitz
        ) {
            this.rebuildLipschitzGrid(false);
        }
        this.rebuildComputeBindGroup();
        if (
            this.experimentState.renderBackend === RenderBackend.marchingCubes
        ) {
            this.rebuildMarchingCubesMesh();
        }
        this.notifyRbfFitChanged();
    }

    onRbfFitChanged(
        listener: (fit: RbfFitResult, config: RbfFitConfig) => void,
    ): () => void {
        this.rbfFitListeners.add(listener);
        if (this.rbfFit) {
            listener(this.rbfFit, this.experimentState.rbfConfig);
        }
        return () => this.rbfFitListeners.delete(listener);
    }

    private notifyRbfFitChanged(): void {
        for (const listener of this.rbfFitListeners) {
            listener(this.rbfFit, this.experimentState.rbfConfig);
        }
    }

    ensureLipschitzGrid(): void {
        if (!this.lipschitzGrid) {
            this.rebuildLipschitzGrid();
        }
    }

    rebuildLipschitzGrid(rebuildBindGroup: boolean = true): void {
        const config: LipschitzGridConfig = {
            resolution:
                this.experimentState.rayMarchingConfig.lipschitzGridResolution,
            samplesPerAxis:
                this.experimentState.rayMarchingConfig.lipschitzSamplesPerAxis,
            safetyFactor:
                this.experimentState.rayMarchingConfig.lipschitzSafetyFactor,
        };
        const start = performance.now();
        const grid = buildLipschitzGrid(
            this.rbfFit,
            this.experimentState.rbfConfig,
            config,
        );
        const elapsed = performance.now() - start;

        this.lipschitzGrid = grid;
        this.lipschitzBuffer.destroy();
        this.lipschitzBuffer = this.createLipschitzBuffer(grid.values);
        if (rebuildBindGroup && this.computeBindGroupLayout) {
            this.rebuildComputeBindGroup();
        }

        console.log(
            `Lipschitz grid ${grid.nx}x${grid.ny}x${grid.nz}: ` +
                `${grid.gradientSampleCount} gradient samples, ` +
                `bounds ${grid.minimumBound}..${grid.maximumBound}, ` +
                `${elapsed} ms`,
        );
    }

    private invalidateLipschitzGrid(): void {
        this.lipschitzGrid = undefined;
        this.lipschitzBuffer.destroy();
        this.lipschitzBuffer = this.createLipschitzBuffer(
            new Float32Array([1]),
        );
    }

    private createLipschitzBuffer(
        values: Float32Array<ArrayBuffer>,
    ): GPUBuffer {
        const buffer = this.device.createBuffer({
            size: values.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(buffer, 0, values);
        return buffer;
    }

    private rebuildComputeBindGroup(): void {
        this.computeBindGroup = this.device.createBindGroup({
            label: "Compute shader bind group",
            layout: this.computeBindGroupLayout,
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
                {
                    binding: 4,
                    resource: { buffer: this.lipschitzBuffer },
                },
                {
                    binding: 5,
                    resource: { buffer: this.metricsWorkBuffer },
                },
            ],
        });
    }
}

function clamp(value: number, min: number, max: number) {
    return Math.min(Math.max(value, min), max);
}
