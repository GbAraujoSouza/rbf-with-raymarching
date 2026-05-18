struct VertexInput {
    @location(0) aPos: vec3f,
    @location(1) color: vec3f,
}

struct FragmentData {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vertexMain(input: VertexInput) -> FragmentData {
    var fragData: FragmentData;
    fragData.position = vec4f(input.aPos, 1.0);
    fragData.color = vec4f(input.color, 1.0);
    return fragData;
}

@fragment
fn fragmentMain(input: FragmentData) -> @location(0) vec4f {
    return input.color;
}
