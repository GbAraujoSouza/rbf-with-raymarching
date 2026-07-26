
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPosition: vec3f,
    @location(1) normal: vec3f,
}

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct MeshUniforms {
    viewProjection: mat4x4f,
    model: mat4x4f,
    cameraPosition: vec4f,
    lightPosition: vec4f,
}

@group(0) @binding(0)
var<uniform> meshUniforms: MeshUniforms;

@vertex
fn main(input: VertexInput) -> VertexOutput {
    let worldPosition = meshUniforms.model * vec4f(input.position, 1.0);
    let worldNormal = meshUniforms.model * vec4f(input.normal, 0.0);

    var output: VertexOutput;
    output.position = meshUniforms.viewProjection * worldPosition;
    output.worldPosition = worldPosition.xyz;
    output.normal= normalize(worldNormal.xyz);
    return output;
}
