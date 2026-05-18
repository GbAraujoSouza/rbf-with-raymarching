import cellShaderCode from "./shaders/cell.wgsl";

const canvas = document.querySelector<HTMLCanvasElement>("#gpu-canvas");
const statusElement = document.querySelector<HTMLParagraphElement>("#status");

if (!canvas || !statusElement) {
    throw new Error("Missing required DOM nodes.");
}

const gpuCanvas = canvas;
const liveStatus = statusElement;

const setStatus = (message: string) => {
    liveStatus.textContent = message;
};

/**
 * Keeps the WebGPU drawing buffer in sync with the canvas' displayed size.
 * The resize uses device pixels so rendering stays sharp on high-DPI screens
 * and avoids redundant writes when the backing buffer is already correct.
 */
const resizeCanvas = () => {
    const devicePixelRatio = window.devicePixelRatio || 1;
    const width = Math.max(
        1,
        Math.floor(gpuCanvas.clientWidth * devicePixelRatio),
    );
    const height = Math.max(
        1,
        Math.floor(gpuCanvas.clientHeight * devicePixelRatio),
    );

    if (gpuCanvas.width !== width || gpuCanvas.height !== height) {
        gpuCanvas.width = width;
        gpuCanvas.height = height;
    }
};

async function initWebGpu() {
    if (!("gpu" in navigator)) {
        setStatus(
            "WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari Technology Preview.",
        );
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) {
        setStatus(
            "The browser supports WebGPU, but no GPU adapter was returned.",
        );
        return;
    }

    const device = await adapter.requestDevice();
    const context = gpuCanvas.getContext("webgpu") as GPUCanvasContext | null;

    if (!context) {
        setStatus("Could not create a WebGPU canvas context.");
        return;
    }

    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

    const configure = () => {
        resizeCanvas();
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

    // Define (x, y) vertices of a square (2 triangles)
    const vertices = new Float32Array([
        -0.4, -0.4, 0.4, -0.4, 0.4, 0.4,

        -0.4, -0.4, 0.4, 0.4, -0.4, 0.4,
    ]);

    const vbo = device.createBuffer({
        label: "cell_vertices",
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vbo, 0, vertices);

    const vboLayout = {
        arrayStride: 8,
        attributes: [
            {
                format: "float32x2",
                offset: 0,
                shaderLocation: 0,
            } as GPUVertexAttribute,
        ],
    };

    // create the shader module with VERTEX and FRAGMENT shader
    // The code is imported from shaders/cell.wgsl
    const cellShaderModule = device.createShaderModule({
        label: "cell_shader",
        code: cellShaderCode,
    });

    const computeShaderModule = device.createShaderModule({
        label: "simulation_compute_shader",
        code: /* wgsl */ `
        @group(0) @binding(0) var<uniform> grid: vec2f;

        @group(0) @binding(1) var<storage, read> cellStateIn: array<u32>;
        @group(0) @binding(2) var<storage, read_write> cellStateOut: array<u32>;

        fn getCellIndex(cell: vec2u) -> u32 {
            return (cell.y % u32(grid.y)) * u32(grid.x) + (cell.x % u32(grid.x));
        }

        fn isCellActive(x: u32, y: u32) -> u32{
            return cellStateIn[ getCellIndex(vec2u(x, y)) ];
        }

        @compute
        @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
        fn computeMain(@builtin(global_invocation_id) cell: vec3u) {
            let activeNeighbors = isCellActive(cell.x+1, cell.y+1) +
                                isCellActive(cell.x+1, cell.y) +
                                isCellActive(cell.x+1, cell.y-1) +
                                isCellActive(cell.x, cell.y-1) +
                                isCellActive(cell.x-1, cell.y-1) +
                                isCellActive(cell.x-1, cell.y) +
                                isCellActive(cell.x-1, cell.y+1) +
                                isCellActive(cell.x, cell.y+1);

            let i = getCellIndex(cell.xy);
            switch activeNeighbors {
                case 2: {
                    cellStateOut[i] = cellStateIn[i]; 
                }
                case 3: {
                    cellStateOut[i] = 1;
                }
                default: {
                    cellStateOut[i] = 0;
                }
            }
        } 
    `,
    });

    // ==== SETUP BUFFERS AND UNIFORMS
    const uniformArray = new Float32Array([GRID_SIZE, GRID_SIZE]);
    const uniformBuffer = device.createBuffer({
        label: "grid_uniform",
        size: uniformArray.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

    const cellStateArrayA = new Uint32Array(GRID_SIZE * GRID_SIZE);
    const cellStateArrayB = new Uint32Array(GRID_SIZE * GRID_SIZE);
    const cellStateStorage = [
        device.createBuffer({
            label: "cell_state_a",
            size: cellStateArrayA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
        device.createBuffer({
            label: "cell_state_b",
            size: cellStateArrayA.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        }),
    ];
    // populate array
    for (let i = 0; i < cellStateArrayA.length; ++i) {
        cellStateArrayA[i] = Math.random() > 0.6 ? 1 : 0;
    }
    device.queue.writeBuffer(cellStateStorage[0], 0, cellStateArrayA);

    for (let i = 0; i < cellStateArrayB.length; i += 2) {
        cellStateArrayB[i] = 1;
    }
    device.queue.writeBuffer(cellStateStorage[1], 0, cellStateArrayB);

    const bindGroupLayout = device.createBindGroupLayout({
        label: "cell_bind_group_layout",
        entries: [
            {
                binding: 0,
                visibility:
                    GPUShaderStage.VERTEX |
                    GPUShaderStage.FRAGMENT |
                    GPUShaderStage.COMPUTE,
                buffer: {},
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage" },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage" },
            },
        ],
    });

    // create bind group with the buffers and arrays
    const bindGroups = [
        device.createBindGroup({
            label: "cell_renderer_bind_group",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: uniformBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: cellStateStorage[0] },
                },
                {
                    binding: 2,
                    resource: { buffer: cellStateStorage[1] },
                },
            ],
        }),
        device.createBindGroup({
            label: "cell_renderer_bind_group_b",
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: { buffer: uniformBuffer },
                },
                {
                    binding: 1,
                    resource: { buffer: cellStateStorage[1] },
                },
                {
                    binding: 2,
                    resource: { buffer: cellStateStorage[0] },
                },
            ],
        }),
    ];

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
            buffers: [vboLayout],
        },
        fragment: {
            module: cellShaderModule,
            entryPoint: "fragmentMain",
            targets: [{ format: presentationFormat }],
        },
    });

    const simulationPipeline = device.createComputePipeline({
        label: "simulation_pipeline",
        layout: pipelineLayout,
        compute: {
            module: computeShaderModule,
            entryPoint: "computeMain",
        },
    });

    const render = () => {
        const encoder = device.createCommandEncoder();

        const computePass = encoder.beginComputePass();

        computePass.setPipeline(simulationPipeline);
        computePass.setBindGroup(0, bindGroups[step % 2]);

        const workgroupCount = Math.ceil(GRID_SIZE / WORKGROUP_SIZE);
        computePass.dispatchWorkgroups(workgroupCount, workgroupCount);

        computePass.end();

        step++;

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

        pass.setBindGroup(0, bindGroups[step % 2]);
        pass.setVertexBuffer(0, vbo);

        pass.draw(vertices.length / 2, GRID_SIZE * GRID_SIZE);

        // end render pass and push command to gpu
        pass.end();
        const commandBuffer = encoder.finish();
        device.queue.submit([commandBuffer]);
    };

    setInterval(render, UPDATE_INTERVAL);
    setStatus("WebGPU initialized successfully.");
}

initWebGpu().catch((error: unknown) => {
    console.error(error);
    const message = error instanceof Error ? error.message : "Unknown error";
    setStatus(`Failed to initialize WebGPU: ${message}`);
});
