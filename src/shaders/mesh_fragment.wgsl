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

fn phong(baseColor: vec3f, point: vec3f, normal: vec3f) -> vec3f {
    let lightPositions = array<vec3f, 3>(
        vec3f(8.0, 10.0, 2.0), // key
        vec3f(-6.0, 4.0, 8.0), // fill
        vec3f(-5.0, 8.0, -8.0), // rim
    );

    let lightIntensities = array<f32, 3>(
        1.0, // key
        0.35, // fill
        0.75, // rim
    );

    let eye = meshUniforms.cameraPosition.xyz;
    let ambient = 0.2 * baseColor;
    let diffuseColor = 0.6 * baseColor;
    let specularColor = vec3f(0.45, 0.45, 0.45);
    let shininess = 48.0;
    let viewDirection = normalize(eye - point);

    var lighting = ambient;
    for (var i = 0u; i < 3u; i++) {
        let lightDirection = normalize(lightPositions[i] - point);
        let reflected = normalize(reflect(-lightDirection, normal));
        let dotLN = max(dot(lightDirection, normal), 0.0);
        let dotRV = max(dot(reflected, viewDirection), 0.0);
        let diffuse = diffuseColor * dotLN;
        let specular = specularColor * pow(dotRV, shininess);

        lighting += lightIntensities[i] * (diffuse + specular);
    }

    return lighting;
}

@fragment
fn main(input: FragmentInput) -> @location(0) vec4f {
    let baseColor = vec3f(0.85, 0.28, 0.18);
    let normal = normalize(input.normal);
    let shaded = phong(baseColor, input.worldPosition, normal);

    return vec4f(shaded, 1.0);
}
