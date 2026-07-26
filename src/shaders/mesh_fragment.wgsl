struct FragmentInput {
    @location(0) worldPosition: vec3f,
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

@fragment
fn main(input: FragmentInput) -> @location(0) vec4f {
    let normal = normalize(input.normal);
    let lightDirection = normalize(meshUniforms.lightPosition.xyz - input.worldPosition);

    let baseColor = vec3f(0.85, 0.28, 0.18);
    let ambient = 0.18 * baseColor;
    let diffuse = max(dot(normal, lightDirection), 0.0) * baseColor;

    return vec4f(ambient + diffuse, 1.0);
}
