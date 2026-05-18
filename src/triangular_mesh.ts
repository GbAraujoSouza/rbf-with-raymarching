export class TriangularMesh {
    buffer: GPUBuffer;
    bufferLayout: GPUVertexBufferLayout;
    vertices: Float32Array<ArrayBuffer>;

    constructor(device: GPUDevice) {
        // prettier-ignore
        this.vertices = new Float32Array(
             [
                 0.0,  0.5, 1.0,  1.0, 0.0, 0.0,
                -0.5, -0.5, 1.0,  0.0, 1.0, 0.0,
                 0.5, -0.5, 1.0,  0.0, 0.0, 1.0
            ]
        );

        const usage: GPUBufferUsageFlags =
            GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST;
        const descriptor: GPUBufferDescriptor = {
            size: this.vertices.byteLength,
            usage: usage,
            //mappedAtCreation: true,
        };

        this.buffer = device.createBuffer(descriptor);
        device.queue.writeBuffer(this.buffer, 0, this.vertices);

        this.bufferLayout = {
            arrayStride: 24,
            attributes: [
                {
                    shaderLocation: 0,
                    format: "float32x3",
                    offset: 0,
                },
                {
                    shaderLocation: 1,
                    format: "float32x3",
                    offset: 12,
                },
            ],
        };
    }
}
