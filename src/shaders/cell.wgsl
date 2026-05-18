struct VertexInput {
    @location(0) aPos: vec2f,
    @builtin(instance_index) instance: u32,
}

struct VertexOutput {
    @builtin(position) aPos: vec4f,
    @location(0) cell: vec2f,
}


@group(0) @binding(0) var<uniform> grid: vec2f;
@group(0) @binding(1) var<storage> cellState: array<u32>;

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
    let i = f32(input.instance);
    let cell = vec2f(i % grid.x, floor(i / grid.x));
    let cellOffset = cell / grid * 2;
    let state = f32(cellState[input.instance]);

    let gridPos = (input.aPos * state + 1) / grid - 1 + cellOffset;

    var output: VertexOutput;
    output.aPos = vec4f(gridPos, 0, 1);
    output.cell = cell;
    return output;
}

struct FragmentInput {
    @location(0) cell: vec2f,
}

@fragment
fn fragmentMain(input: FragmentInput) -> @location(0) vec4f {
    let c = input.cell/grid;
    return vec4f(c, 1-c.x, 1);
}
