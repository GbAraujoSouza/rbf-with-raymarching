struct VertexInput {
    @location(0) aPos: vec3f,
    @location(1) textureCoord: vec2f,
}

struct FragmentData {
    @builtin(position) position: vec4f,
    @location(0) textureCoord: vec2f,
}

@vertex
fn vertMain(@builtin(vertex_index) index: u32) -> FragmentData {
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


    var fragData: FragmentData;
    fragData.position = vec4f(positions[index], 0.0, 1.0);
    fragData.textureCoord = texCoords[index];

    return fragData;
}

@group(0) @binding(0) var screenSampler: sampler;
@group(0) @binding(1) var colorBuffer: texture_2d<f32>;

@fragment
fn fragMain(input: FragmentData) -> @location(0) vec4f {
    return textureSample(colorBuffer, screenSampler, input.textureCoord);
}
