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
    var positions = array<vec2f, 6>(
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0)
    );

    var texCoords = array<vec2f, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0)
    );


    var output: VertexOutput;
    output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
    output.uv = texCoords[vertexIndex];

    return output;
}

@group(0) @binding(0) var screenSampler: sampler;
@group(0) @binding(1) var colorBuffer: texture_2d<f32>;


@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return textureSample(colorBuffer, screenSampler, input.uv);
}
