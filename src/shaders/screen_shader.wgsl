struct SceneUniforms {
    screenAndCounts: vec4f, // width, height, # points, show_debug 
    cameraPosition: vec4f,
    cameraForward: vec4f,
    cameraRight: vec4f,
    cameraUp: vec4f,
    marchParams: vec4f,
    rbfParams: vec4f,
    lightPosition: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}


@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0),
    );

    var output: VertexOutput;
    let position = positions[vertexIndex];
    output.position = vec4f(position, 0.0, 1.0);
    output.uv = position * 0.5 + vec2f(0.5, 0.5);
    return output;
}

@group(0) @binding(0) var screenSampler: sampler;
@group(0) @binding(1) var colorBuffer: texture_2d<f32>;


@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return textureSample(colorBuffer, screenSampler, input.uv);
}
