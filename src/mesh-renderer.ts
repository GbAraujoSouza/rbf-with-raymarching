import meshVertexCode from "./shaders/mesh_vertex.wgsl";
import meshFragmentCode from "./shaders/mesh_fragment.wgsl";
import { ExtractedMesh } from "./mesh";
import { Mat4, Vec3 } from "wgpu-matrix";

export class MeshRenderer {
    static readonly DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

    device: GPUDevice;
    format: GPUTextureFormat;

    pipeline!: GPURenderPipeline;
    uniformBuffer!: GPUBuffer;
    positionsBuffer!: GPUBuffer;
    normalsBuffer!: GPUBuffer;

    bindGroup!: GPUBindGroup;
    depthTexture!: GPUTexture;
    depthTextureView!: GPUTextureView;

    vertexCount!: number;
    width: number = 0;
    height: number = 0;

    constructor(device: GPUDevice, format: GPUTextureFormat) {
        this.device = device;
        this.format = format;
    }

    initialize(): void {
        this.uniformBuffer = this.device.createBuffer({
            size: 40 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" },
                },
                // {
                //     binding: 1,
                //     visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                //     buffer: { type: "read-only-storage" },
                // },
                // {
                //     binding: 2,
                //     visibility: GPUShaderStage.FRAGMENT | GPUShaderStage.VERTEX,
                //     buffer: { type: "read-only-storage" },
                // },
            ],
        });

        // this.bindGroup = this.device.createBindGroup({
        //     layout: bindGroupLayout,
        //     entries: [
        //         {
        //             binding: 0,
        //             resource: { buffer: this.uniformBuffer },
        //         },
        //         {
        //             binding: 1,
        //             resource: this.positionsBuffer,
        //         },
        //         {
        //             binding: 2,
        //             resource: this.normalsBuffer,
        //         },
        //     ],
        // });

        const meshVertexModule = this.device.createShaderModule({
            code: meshVertexCode,
        });

        const meshFragmentModule = this.device.createShaderModule({
            code: meshFragmentCode,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        this.pipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: meshVertexModule,
                entryPoint: "main",
                buffers: [
                    {
                        arrayStride: 3 * 4,
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: "float32x3",
                            },
                        ],
                    },
                    {
                        arrayStride: 3 * 4,
                        attributes: [
                            {
                                shaderLocation: 1,
                                offset: 0,
                                format: "float32x3",
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: meshFragmentModule,
                entryPoint: "main",
                targets: [{ format: this.format }],
            },

            primitive: {
                topology: "triangle-list",
            },
            depthStencil: {
                format: MeshRenderer.DEPTH_FORMAT,
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });
    }

    setMesh(mesh: ExtractedMesh) {
        // this.positionsBuffer.destroy();
        // this.normalsBuffer.destroy();

        const usage: GPUBufferUsageFlags =
            GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;

        this.positionsBuffer = this.device.createBuffer({
            size: mesh.positions.byteLength,
            usage: usage,
        });
        this.device.queue.writeBuffer(this.positionsBuffer, 0, mesh.positions);

        this.normalsBuffer = this.device.createBuffer({
            size: mesh.normals.byteLength,
            usage: usage,
        });
        this.device.queue.writeBuffer(this.normalsBuffer, 0, mesh.normals);

        // this.positionsBuffer = this.device.createBuffer({
        //     size: mesh.positions.byteLength,
        //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        // });
        // this.normalsBuffer = this.device.createBuffer({
        //     size: mesh.normals.byteLength,
        //     usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        // });

        // this.device.queue.writeBuffer(this.positionsBuffer, 0, mesh.positions);
        // this.device.queue.writeBuffer(this.normalsBuffer, 0, mesh.normals);

        this.bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformBuffer },
                },
                // {
                //     binding: 1,
                //     resource: this.positionsBuffer,
                // },
                // {
                //     binding: 2,
                //     resource: this.normalsBuffer,
                // },
            ],
        });

        this.vertexCount = mesh.positions.length / 3;
    }

    resize(width: number, height: number): void {
        if (this.width === width && this.height === height) {
            return;
        }

        this.width = width;
        this.height = height;

        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            size: { width, height },
            format: MeshRenderer.DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthTextureView = this.depthTexture.createView();
    }

    render(pass: GPURenderPassEncoder, uniforms: MeshCameraUniformInput): void {
        const uniformArrayBuffer = new Float32Array(40);

        uniformArrayBuffer.set(uniforms.viewProjection, 0);
        uniformArrayBuffer.set(uniforms.model, 16);

        uniformArrayBuffer.set(
            [
                uniforms.cameraPosition[0],
                uniforms.cameraPosition[1],
                uniforms.cameraPosition[2],
                0,
            ],
            32,
        );

        uniformArrayBuffer.set(
            [
                uniforms.lightPosition[0],
                uniforms.lightPosition[1],
                uniforms.lightPosition[2],
                0,
            ],
            36,
        );

        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            uniformArrayBuffer,
        );

        // draw stuff
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.setVertexBuffer(0, this.positionsBuffer);
        pass.setVertexBuffer(1, this.normalsBuffer);
        pass.draw(this.vertexCount);

        pass.end();
        //const commandBuffer = encoder.finish();
        //this.device.queue.submit([commandBuffer]);
    }

    destroy(): void {
        this.depthTexture?.destroy();
    }
}

export interface MeshCameraUniformInput {
    viewProjection: Mat4;
    model: Mat4;
    cameraPosition: Vec3;
    lightPosition: Vec3;
}
