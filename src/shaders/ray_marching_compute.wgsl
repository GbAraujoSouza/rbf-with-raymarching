struct SceneData {
    cameraPosition: vec3f,
    cameraForward: vec3f,
    cameraRight: vec3f,
    cameraUp: vec3f,
}

struct SceneUniforms {
    screenAndCounts: vec4f, // width, height, # points, show_debug 
    cameraPosition: vec3f,
    cameraForward: vec3f,
    cameraRight: vec3f,
    cameraUp: vec3f,
    marchParams: vec4f,
    rbfParams: vec4f,
    lightPosition: vec4f,
}


@group(0) @binding(0) var screenTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> sceneUniforms: SceneUniforms;
@group(0) @binding(2) var<storage, read> samplePositions: SamplePositions;
@group(0) @binding(3) var<storage, read> sampleWeights: SampleWeights;

struct SamplePositions {
    values: array<vec4f>,
}

struct SampleWeights {
    values: array<f32>,
}


const MAX_MARCHING_STEPS = 255;

fn gaussian(radius: f32) -> f32 {
    let epsilon = sceneUniforms.rbfParams.x;
    let scaled = epsilon * radius;
    return exp(-(scaled * scaled));
}

fn rbfField(point: vec3f) -> f32 {
    let sampleCount = i32(sceneUniforms.screenAndCounts.z);
    var total = 0.0;

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let weight = sampleWeights.values[index];
        total += gaussian(distance(point, center)) * weight;
    }

    return total;
}

fn pointsSdf(point: vec3f) -> f32 {
    let sampleCount = i32(sceneUniforms.screenAndCounts.z);
    var minimumDistance = sceneUniforms.marchParams.z;

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let distanceToPoint = distance(point, center) - sceneUniforms.rbfParams.y;
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
    let maxDistance = sceneUniforms.marchParams.z;
    let epsilon = sceneUniforms.marchParams.w;
    let correctionPower = sceneUniforms.marchParams.x;
    let correctionLinear = sceneUniforms.marchParams.y;
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
    let epsilon = sceneUniforms.marchParams.w;
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
    let lightPosition = sceneUniforms.lightPosition.xyz;
    let lightIntensity = vec3f(1.0, 1.0, 1.0);
    let eye = sceneUniforms.cameraPosition.xyz;
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

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationId: vec3u) {

    let screenSize: vec2i = vec2i(textureDimensions(screenTexture));
    let screenPos: vec2i = vec2i(i32(GlobalInvocationId.x), i32(GlobalInvocationId.y));
    if (screenPos.x >= screenSize.x || screenPos.y >= screenSize.y) {
        return;
    }

    let horizontalCoeff: f32 = (f32(screenPos.x) - f32(screenSize.x) / 2.0) / f32(screenSize.x);
    let verticalCoeff: f32 = (f32(screenPos.y) - f32(screenSize.y) / 2.0) / f32(screenSize.x);


    let rayOrigin = sceneUniforms.cameraPosition.xyz;
    let rayDirection = normalize(sceneUniforms.cameraForward + horizontalCoeff * sceneUniforms.cameraRight + verticalCoeff * sceneUniforms.cameraUp);
    //let rayDir = rayDirection(input.uv);
    let showPoints = sceneUniforms.screenAndCounts.w > 0.5;

    let rbfDistance = shortestDistanceToSurface(rayOrigin, rayDirection, false);
    var hitDistance = rbfDistance;
    var hitPoints = false;

    if (showPoints) {
        let pointsDistance = shortestDistanceToSurface(rayOrigin, rayDirection, true);
        if (pointsDistance < hitDistance) {
            hitDistance = pointsDistance;
            hitPoints = true;
        }
    }

    if (hitDistance >= sceneUniforms.marchParams.z) {
        textureStore(screenTexture, screenPos, vec4f(0.5, 0.5, 0.5, 1.0));
    }

    let point = rayOrigin + hitDistance * rayDirection;
    let normal = estimateNormal(point, hitPoints);
    let baseColor = select(vec3f(0.93, 0.32, 0.25), vec3f(0.96, 0.84, 0.23), hitPoints);
    let shaded = phong(baseColor, point, normal);

    //return vec4f(shaded, 1.0);
    textureStore(screenTexture, screenPos, vec4f(shaded, 1.0));
}