import { glMatrix, mat4 } from "gl-matrix";
import screenShaderCode from "./shaders/screen_shader.wgsl";
import raytracerShaderCode from "./shaders/raytracing_shader.wgsl";
import { TriangularMesh } from "./triangular_mesh";
import { Scene } from "./scene";

export class Renderer {
    canvas: HTMLCanvasElement;

    adapter!: GPUAdapter;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;

    scene: Scene;

    // Pipeline objects
    rayTracingBindGroup!: GPUBindGroup;
    rayTracingPipeline!: GPUComputePipeline;

    screenBindGroup!: GPUBindGroup;
    screenPipeline!: GPURenderPipeline;

    // assets
    colorBuffer!: GPUTexture;
    colorBufferView!: GPUTextureView;
    sampler!: GPUSampler;
    sceneParameters!: GPUBuffer;
    objectsBuffer!: GPUBuffer;

    // Metrics
    frameCount: number = 0;
    lastFpsTime: number = performance.now();

    constructor(canvas: HTMLCanvasElement, scene: Scene) {
        this.canvas = canvas;
        this.scene = scene;
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

    /**
     * there are some artifacts in here:
     * 1. Screen
     *      1.1 bind group Layout
     *      1.2 bind group
     *      1.3 pipeline Layout
     *      1.4 render pipeline
     * 2. Compute (ray tracing)
     *      2.1 bind group Layout
     *      2.2 bind group
     *      2.3 pipeline Layout
     *      3.4 compute pipeline
     */
    async makePipeline() {
        const rayTracingBindGroupLayout = this.device.createBindGroupLayout({
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
                    buffer: {
                        type: "uniform",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
            ],
        });

        this.rayTracingBindGroup = this.device.createBindGroup({
            layout: rayTracingBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.colorBufferView,
                },
                {
                    binding: 1,
                    resource: this.sceneParameters,
                },
                {
                    binding: 2,
                    resource: this.objectsBuffer,
                },
            ],
        });

        const rayTracingPipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [rayTracingBindGroupLayout],
        });

        const rayTracingModule = this.device.createShaderModule({
            code: raytracerShaderCode,
        });

        const rtInfo = await rayTracingModule.getCompilationInfo();
        if (rtInfo.messages.length > 0) {
            console.warn("RayTracing Shader Messages:");
            rtInfo.messages.forEach((m) =>
                console.warn(
                    `${m.type} at ${m.lineNum}:${m.linePos} - ${m.message}`,
                ),
            );
        }

        this.rayTracingPipeline = this.device.createComputePipeline({
            layout: rayTracingPipelineLayout,
            compute: {
                module: rayTracingModule,
                entryPoint: "main",
            },
        });

        const screenBindGroupLayout = this.device.createBindGroupLayout({
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

        this.screenBindGroup = this.device.createBindGroup({
            layout: screenBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: this.sampler,
                },
                {
                    binding: 1,
                    resource: this.colorBufferView,
                },
            ],
        });

        const screenPipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [screenBindGroupLayout],
        });

        const screenModule = this.device.createShaderModule({
            code: screenShaderCode,
        });

        const screenInfo = await screenModule.getCompilationInfo();
        if (screenInfo.messages.length > 0) {
            console.warn("Screen Shader Messages:");
            screenInfo.messages.forEach((m) =>
                console.warn(
                    `${m.type} at ${m.lineNum}:${m.linePos} - ${m.message}`,
                ),
            );
        }

        this.screenPipeline = this.device.createRenderPipeline({
            layout: screenPipelineLayout,
            vertex: {
                module: screenModule,
                entryPoint: "vertMain",
            },
            fragment: {
                module: screenModule,
                entryPoint: "fragMain",
                targets: [
                    {
                        format: this.format,
                    },
                ],
            },
            primitive: { topology: "triangle-list" },
        });
    }

    createAssets() {
        this.colorBuffer = this.device.createTexture({
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

        this.colorBufferView = this.colorBuffer.createView();

        const sampleDescriptor: GPUSamplerDescriptor = {
            addressModeU: "repeat",
            addressModeV: "repeat",
            magFilter: "linear",
            minFilter: "nearest",
            mipmapFilter: "nearest",
            maxAnisotropy: 1,
        };

        this.sampler = this.device.createSampler(sampleDescriptor);

        const parameterBufferDescriptor: GPUBufferDescriptor = {
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        };
        this.sceneParameters = this.device.createBuffer(
            parameterBufferDescriptor,
        );

        const objectsBufferDescriptor: GPUBufferDescriptor = {
            // the sphere object data has pos(12bytes) + 1byte padding + color(12bytes) + radius(4byte)
            size: 32 * this.scene.spheres.length,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        };
        this.objectsBuffer = this.device.createBuffer(objectsBufferDescriptor);
    }

    prepareScene() {
        const sceneData = {
            cameraPos: this.scene.camera.position,
            cameraForward: this.scene.camera.forward,
            cameraRight: this.scene.camera.right,
            cameraUp: this.scene.camera.up,
            sphereCount: this.scene.spheres.length,
        };
        this.device.queue.writeBuffer(
            this.sceneParameters,
            0,
            new Float32Array([
                sceneData.cameraPos[0],
                sceneData.cameraPos[1],
                sceneData.cameraPos[2],
                0.0,
                sceneData.cameraForward[0],
                sceneData.cameraForward[1],
                sceneData.cameraForward[2],
                0.0,
                sceneData.cameraRight[0],
                sceneData.cameraRight[1],
                sceneData.cameraRight[2],
                0.0,
                sceneData.cameraUp[0],
                sceneData.cameraUp[1],
                sceneData.cameraUp[2],
                sceneData.sphereCount,
            ]),
            0,
            16,
        );

        const sphereData: Float32Array = new Float32Array(
            8 * this.scene.spheres.length,
        );
        for (let i = 0; i < this.scene.spheres.length; i++) {
            sphereData[8 * i] = this.scene.spheres[i].center[0];
            sphereData[8 * i + 1] = this.scene.spheres[i].center[1];
            sphereData[8 * i + 2] = this.scene.spheres[i].center[2];
            sphereData[8 * i + 3] = 0.0;
            sphereData[8 * i + 4] = this.scene.spheres[i].color[0];
            sphereData[8 * i + 5] = this.scene.spheres[i].color[1];
            sphereData[8 * i + 6] = this.scene.spheres[i].color[2];
            sphereData[8 * i + 7] = this.scene.spheres[i].radius;
        }

        this.device.queue.writeBuffer(
            this.objectsBuffer,
            0,
            <ArrayBuffer>(<unknown>sphereData),
            0,
            8 * this.scene.spheres.length,
        );
    }

    render = () => {
        this.prepareScene();

        let start: number = performance.now();
        const encoder = this.device.createCommandEncoder();

        const rayTracingPass: GPUComputePassEncoder =
            encoder.beginComputePass();
        rayTracingPass.setBindGroup(0, this.rayTracingBindGroup);
        rayTracingPass.setPipeline(this.rayTracingPipeline);
        rayTracingPass.dispatchWorkgroups(
            Math.ceil(this.canvas.width / 8),
            Math.ceil(this.canvas.height / 8),
            1,
        );
        rayTracingPass.end();

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
        pass.setPipeline(this.screenPipeline);

        pass.setBindGroup(0, this.screenBindGroup);

        pass.draw(6, 1, 0, 0);

        pass.end();

        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        this.device.queue.onSubmittedWorkDone().then(() => {
            let end: number = performance.now();
            const renderTimeLabel: HTMLElement = <HTMLElement>(
                document.querySelector("#render-time-label")
            );
            if (renderTimeLabel) {
                renderTimeLabel.innerText = (end - start).toFixed(2);
            }

            this.frameCount++;
            // Atualiza a cada 1 segundo (1000 milissegundos)
            if (end - this.lastFpsTime >= 1000.0) {
                const fpsLabel: HTMLElement = <HTMLElement>(
                    document.querySelector("#fps-label")
                );
                if (fpsLabel) {
                    fpsLabel.innerText = this.frameCount.toString();
                }
                this.frameCount = 0;
                this.lastFpsTime = end;
            }
        });

        requestAnimationFrame(this.render);
    };
}
