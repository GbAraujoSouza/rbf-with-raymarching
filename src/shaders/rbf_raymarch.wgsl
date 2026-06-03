struct SceneUniforms {
    screenAndCounts: vec4f,
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

struct SamplePositions {
    values: array<vec4f>,
}

struct SampleWeights {
    values: array<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: SceneUniforms;
@group(0) @binding(1) var<storage, read> samplePositions: SamplePositions;
@group(0) @binding(2) var<storage, read> sampleWeights: SampleWeights;

const MAX_MARCHING_STEPS = 255;

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

fn gaussian(radius: f32) -> f32 {
    let epsilon = uniforms.rbfParams.x;
    let scaled = epsilon * radius;
    return exp(-(scaled * scaled));
}

fn rbfField(point: vec3f) -> f32 {
    let sampleCount = i32(uniforms.screenAndCounts.z);
    var total = 0.0;

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let weight = sampleWeights.values[index];
        total += gaussian(distance(point, center)) * weight;
    }

    return total;
}

fn pointsSdf(point: vec3f) -> f32 {
    let sampleCount = i32(uniforms.screenAndCounts.z);
    var minimumDistance = uniforms.marchParams.z;

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let distanceToPoint = distance(point, center) - uniforms.rbfParams.y;
        minimumDistance = min(minimumDistance, distanceToPoint);
    }

    return minimumDistance;
}

fn sceneSdf(point: vec3f, usePoints: bool) -> f32 {
    if (usePoints) {
        return pointsSdf(point);
    }

    return rbfField(point);
}

fn shortestDistanceToSurface(rayOrigin: vec3f, rayDirection: vec3f, usePoints: bool) -> f32 {
    let maxDistance = uniforms.marchParams.z;
    let epsilon = uniforms.marchParams.w;
    let correctionPower = uniforms.marchParams.x;
    let correctionLinear = uniforms.marchParams.y;
    var depth = 0.0;

    for (var step = 0; step < MAX_MARCHING_STEPS; step += 1) {
        let point = rayOrigin + depth * rayDirection;
        let fieldValue = sceneSdf(point, usePoints);
        let distanceToSurface = abs(fieldValue);

        if (usePoints) {
            if (distanceToSurface < epsilon) {
                return depth;
            }
        } else {
            if (fieldValue < 0.0 && length(point) < 2) {
                return depth;
            }
        }

        var stepDistance = distanceToSurface;
        if (!usePoints) {

            // make the correcion to the step
            stepDistance = correctionLinear * pow(max(distanceToSurface, epsilon), correctionPower);
            
            let distToBounding = length(point) - 2;
            if (distToBounding > 0.0) {
                stepDistance = max(stepDistance, distToBounding);
            }
        }

        depth += max(stepDistance, epsilon);

        if (depth >= maxDistance) {
            return maxDistance;
        }
    }

    return maxDistance;
}

fn estimateNormal(point: vec3f, usePoints: bool) -> vec3f {
    let epsilon = uniforms.marchParams.w;
    let xOffset = vec3f(epsilon, 0.0, 0.0);
    let yOffset = vec3f(0.0, epsilon, 0.0);
    let zOffset = vec3f(0.0, 0.0, epsilon);

    let gradient = vec3f(
        sceneSdf(point + xOffset, usePoints) - sceneSdf(point - xOffset, usePoints),
        sceneSdf(point + yOffset, usePoints) - sceneSdf(point - yOffset, usePoints),
        sceneSdf(point + zOffset, usePoints) - sceneSdf(point - zOffset, usePoints),
    );

    return normalize(gradient);
}

fn phong(baseColor: vec3f, point: vec3f, normal: vec3f) -> vec3f {
    let lightPosition = uniforms.lightPosition.xyz;
    let lightIntensity = vec3f(1.0, 1.0, 1.0);
    let eye = uniforms.cameraPosition.xyz;
    let ambient = 0.2 * baseColor;
    let diffuseColor = 0.6 * baseColor;
    let specularColor = vec3f(0.45, 0.45, 0.45);
    let shininess = 48.0;

    let lightDirection = normalize(lightPosition - point);
    let viewDirection = normalize(eye - point);
    let reflected = normalize(reflect(-lightDirection, normal));

    let dotLN = max(dot(lightDirection, normal), 0.0);
    let dotRV = max(dot(reflected, viewDirection), 0.0);

    let diffuse = diffuseColor * dotLN;
    let specular = specularColor * pow(dotRV, shininess);

    return lightIntensity * (ambient + diffuse + specular);
}

fn rayDirection(uv: vec2f) -> vec3f {
    let aspect = uniforms.screenAndCounts.x / uniforms.screenAndCounts.y;
    let halfFovTangent = tan(0.5 * uniforms.rbfParams.z);
    let ndc = vec2f(
        uv.x * 2.0 - 1.0,
        1.0 - uv.y * 2.0,
    );

    let ray =
        uniforms.cameraForward.xyz +
        ndc.x * aspect * halfFovTangent * uniforms.cameraRight.xyz +
        ndc.y * halfFovTangent * uniforms.cameraUp.xyz;

    return normalize(ray);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    let rayOrigin = uniforms.cameraPosition.xyz;
    let rayDir = rayDirection(input.uv);
    let showPoints = uniforms.screenAndCounts.w > 0.5;

    let rbfDistance = shortestDistanceToSurface(rayOrigin, rayDir, false);
    var hitDistance = rbfDistance;
    var hitPoints = false;

    if (showPoints) {
        let pointsDistance = shortestDistanceToSurface(rayOrigin, rayDir, true);
        if (pointsDistance < hitDistance) {
            hitDistance = pointsDistance;
            hitPoints = true;
        }
    }

    if (hitDistance >= uniforms.marchParams.z) {
        discard;
    }

    let point = rayOrigin + hitDistance * rayDir;
    let normal = estimateNormal(point, hitPoints);
    let baseColor = select(vec3f(0.93, 0.32, 0.25), vec3f(0.96, 0.84, 0.23), hitPoints);
    let shaded = phong(baseColor, point, normal);

    return vec4f(shaded, 1.0);
}
