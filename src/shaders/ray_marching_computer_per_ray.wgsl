struct SceneUniforms {
    screenAndCounts: vec4f, // width, height, # points, show_debug 
    cameraPosition: vec4f,
    cameraForward: vec4f,
    cameraRight: vec4f,
    cameraUp: vec4f,
    
    // RAY MARCHING PARAMS
    correctionPower: f32,
    correctionLinear: f32,
    maxDistance: f32,
    /*
        This epsilon is used for 2 things:
        1. Check if the ray is too close to the surface
        2. Calculate normal near the point (small epsilon shift)
    */
    epsilon: f32,

    // RBF PARAMS
    gaussianEpsilon: f32,
    debugPointRadius: f32,
    stepStrategy: f32, //later transform to u32
    renderMode: f32, //later transform to u32

    lightPosition: vec3f,
    kernelType: f32,
    boxMin: vec4f,
    boxMax: vec4f,
    worldToObject: mat4x4f,

    maxSteps: f32,
    fieldOfView: f32,
    lipschitzGridDimensions: vec4f,
}


@group(0) @binding(0) var screenTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> sceneUniforms: SceneUniforms;
@group(0) @binding(2) var<storage, read> samplePositions: SamplePositions;
@group(0) @binding(3) var<storage, read> sampleWeights: SampleWeights;
@group(0) @binding(4) var<storage, read> lipschitzValues: LipschitzValues;
@group(0) @binding(5) var<storage, read_write> metricsWorkBuffer: array<f32>;

struct SamplePositions {
    values: array<vec4f>,
}

struct SampleWeights {
    values: array<f32>,
}

struct LipschitzValues {
    values: array<f32>,
}

struct MarchingResult {
    distance: f32,
    stepCount: u32,
}

struct CellLocalLipschitzSteps {
    surface: f32,
    cellExitDepth: f32,
}

struct LipschitzCellAxis {
    coordinate: u32,
    exitDepth: f32,
}

const RAY_LANES: u32 = 32u;
const RAY_MARCHING: u32 = 0u;
const RAY_HIT: u32 = 1u;
const RAY_MISS: u32 = 2u;
const STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ: u32 = 2u;
const MINIMUM_LIPSCHITZ_BOUND: f32 = 1e-4;
const DIRECTION_EPSILON: f32 = 1e-8;
const LARGE_DISTANCE: f32 = 1e30;

struct RayState {
    depth: f32,
    distance: f32,
    stepCount: u32,
    status: u32,
}

var<workgroup> partialSums: array<f32, RAY_LANES>;
var<workgroup> sharedRayState: RayState;

fn gaussian(point: vec3f, center: vec3f) -> f32 {
    let epsilon = sceneUniforms.gaussianEpsilon;
    let epsilonSquared = epsilon * epsilon;
    let d = point - center;
    return exp(-epsilonSquared * dot(d, d));
}

fn kernel(point: vec3f, center: vec3f) -> f32 {
    switch(u32(sceneUniforms.kernelType)) {
        case 0u {
            // linear
            return length(point - center);
        }
        case 1u {
            // gaussian
            return gaussian(point, center);
        }
        case 2u {
            // cubic
            let radius = length(point - center);
            return radius * radius * radius;
        }
        case 3u {
            // quintic
            let radius = length(point - center);
            let r2 = radius * radius;
            return r2 * r2 * radius;
        }
        case 4u {
            // thin plate
            // equation r^2 * ln(r) = |d|^2 * 0.5 * ln(d)
            // where d is the offset. working with dot() operator is less expensive
            let d: vec3f = point - center;
            let q: f32 = dot(d, d);
            if (q == 0.0) {
                return 0.0;
            }
            return q * 0.5 * log(q);
        }
        default {
            // default to lenear
            return length(point - center);
        }
    }
}

fn rbfGradient(point: vec3f) -> vec3f {
    let sampleCount = i32(sceneUniforms.screenAndCounts.z);
    var gradient = vec3f(0.0, 0.0, 0.0);

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let offset = point - center;
        //let radius = length(offset);
        let radius = 1.0;
        var derivativeScale = 0.0;

        switch(u32(sceneUniforms.kernelType)) {
            case 0u {
                if (radius > 0.0) {
                    derivativeScale = 1.0 / radius;
                }
            }
            case 1u {
                let epsilon = sceneUniforms.gaussianEpsilon;
                let epsilonSquared = epsilon * epsilon;
                derivativeScale =
                    -2.0 *
                    epsilonSquared *
                    exp(-epsilonSquared * radius * radius);
            }
            case 2u {
                derivativeScale = 3.0 * radius;
            }
            case 3u {
                derivativeScale = 5.0 * radius * radius * radius;
            }
            case 4u {
                if (radius > 0.0) {
                    let q = dot(offset, offset);
                    derivativeScale = (log(q) + 1);
                }
            }
            default {
                if (radius > 0.0) {
                    derivativeScale = 1.0 / radius;
                }
            }
        }

        gradient += sampleWeights.values[index] * derivativeScale * offset;
    }

    return gradient;
}

fn rbfField(point: vec3f, lane: u32) -> f32 {
    let sampleCount = u32(sceneUniforms.screenAndCounts.z);
    var partial = 0.0;

    for (var index = lane; index < sampleCount; index += RAY_LANES) {
        let center = samplePositions.values[index].xyz;
        let weight = sampleWeights.values[index];
        partial += kernel(point, center) * weight;
    }

    partialSums[lane] = partial;
    workgroupBarrier();

    // Combine 32 partial sums in five parallel stages.
    for (var stride = RAY_LANES / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            partialSums[lane] += partialSums[lane + stride];
        }
        workgroupBarrier();
    }

    // Broadcast the result before any lane reuses the scratch array.
    return workgroupUniformLoad(&partialSums[0]);
}

fn adjustedStep(baseStep: f32) -> f32 {
    let epsilon = sceneUniforms.epsilon;
    let linear = sceneUniforms.correctionLinear;
    let power = sceneUniforms.correctionPower;
    let flooredBase = max(baseStep, epsilon);
    if (linear == 1.0 && power == 1.0) {
        return flooredBase;
    }
    return max(linear * pow(flooredBase, power), epsilon);
}

fn calculateStep(distanceToSurface: f32, point: vec3f) -> f32 {
    let strategyId = u32(sceneUniforms.stepStrategy);
    switch strategyId {
        case 1u {
            // Gradient
            let gradient = rbfGradient(point);
            return adjustedStep(distanceToSurface / max(length(gradient), 0.01));
        }
        case 3u {
            // Step Tracing
            return 0.01;
        }
        default {
            // Naive
            return adjustedStep(distanceToSurface);
        }
    }
}

fn lipschitzGridDimensions() -> vec3u {
    return vec3u(sceneUniforms.lipschitzGridDimensions.xyz);
}

fn lipschitzCellAxis(
    point: f32,
    rayOrigin: f32,
    rayDirection: f32,
    depth: f32,
    boxMin: f32,
    boxMax: f32,
    dimension: u32,
) -> LipschitzCellAxis {
    let extent = boxMax - boxMin;
    let cellSize = extent / f32(dimension);
    let scaledPosition =
        clamp((point - boxMin) / extent, 0.0, 1.0) * f32(dimension);
    if (abs(rayDirection) <= DIRECTION_EPSILON) {
        return LipschitzCellAxis(min(u32(scaledPosition), dimension - 1u), LARGE_DISTANCE);
    }

    // Use the position to find a nearby face, then resolve its side in ray depth.
    let nearestFace = i32(round(scaledPosition));
    let invDirection = 1.0 / rayDirection;
    let facePosition = boxMin + f32(nearestFace) * cellSize;
    let faceDepth = (facePosition - rayOrigin) * invDirection;
    let forward = rayDirection > 0.0;
    let reachedFace = depth >= faceDepth;
    let upperCell = reachedFace == forward;
    let coordinate = u32(clamp(nearestFace - select(1, 0, upperCell), 0, i32(dimension) - 1));

    let exitFace = coordinate + select(0u, 1u, forward);
    let exitPosition = select(boxMin + f32(exitFace) * cellSize, boxMax, exitFace == dimension);
    return LipschitzCellAxis(coordinate, (exitPosition - rayOrigin) * invDirection);
}

fn lipschitzCellIndex(cell: vec3u) -> u32 {
    let dimensions = lipschitzGridDimensions();
    return cell.z * dimensions.x * dimensions.y +
        cell.y * dimensions.x +
        cell.x;
}

fn localLipschitzBound(cell: vec3u) -> f32 {
    return max(
        lipschitzValues.values[lipschitzCellIndex(cell)],
        MINIMUM_LIPSCHITZ_BOUND,
    );
}

fn cellLocalLipschitzSteps(
    distanceToSurface: f32,
    point: vec3f,
    rayOrigin: vec3f,
    rayDirection: vec3f,
    depth: f32,
) -> CellLocalLipschitzSteps {
    let dimensions = lipschitzGridDimensions();
    let boxMin = sceneUniforms.boxMin.xyz;
    let boxMax = sceneUniforms.boxMax.xyz;
    let x = lipschitzCellAxis(point.x, rayOrigin.x, rayDirection.x, depth, boxMin.x, boxMax.x, dimensions.x);
    let y = lipschitzCellAxis(point.y, rayOrigin.y, rayDirection.y, depth, boxMin.y, boxMax.y, dimensions.y);
    let z = lipschitzCellAxis(point.z, rayOrigin.z, rayDirection.z, depth, boxMin.z, boxMax.z, dimensions.z);
    let cell = vec3u(x.coordinate, y.coordinate, z.coordinate);
    let surfaceStep = distanceToSurface / localLipschitzBound(cell);
    return CellLocalLipschitzSteps(surfaceStep, min(x.exitDepth, min(y.exitDepth, z.exitDepth)));
}

struct RayBoxIntercept {
    hit: bool,
    tMin: f32,
    tMax: f32,
}

fn intersectAABB(rayOrigin: vec3f, rayDirection: vec3f, boxMin: vec3f, boxMax: vec3f) -> RayBoxIntercept {
    let invDir: vec3f = 1.0 / rayDirection;

    let t0: vec3f = (boxMin - rayOrigin) * invDir;
    let t1: vec3f = (boxMax - rayOrigin) * invDir;

    let tNear: vec3f = min(t0, t1);
    let tFar: vec3f = max(t0, t1);

    let tMin: f32 = max(max(tNear.x, tNear.y), tNear.z);
    let tMax: f32 = min(min(tFar.x, tFar.y), tFar.z);

    return RayBoxIntercept(tMax >= max(tMin, 0.0), tMin, tMax);
}

fn shortestDistanceToSurface(
    rayOrigin: vec3f,
    rayDirection: vec3f,
    lane: u32,
) -> MarchingResult {
    let maxDistance = sceneUniforms.maxDistance;
    let epsilon = sceneUniforms.epsilon;
    let maxSteps = u32(sceneUniforms.maxSteps);
    let localRayOrigin = (sceneUniforms.worldToObject * vec4f(rayOrigin, 1.0)).xyz;
    let localRayDirection = (sceneUniforms.worldToObject * vec4f(rayDirection, 0.0)).xyz;
    let hitBox = intersectAABB(
        localRayOrigin,
        localRayDirection,
        sceneUniforms.boxMin.xyz,
        sceneUniforms.boxMax.xyz,
    );
    let endDepth = min(hitBox.tMax, maxDistance);

    if (lane == 0u) {
        sharedRayState = RayState(max(hitBox.tMin, 0.0), maxDistance, 0u, RAY_MISS);
        if (hitBox.hit) {
            sharedRayState.status = RAY_MARCHING;
        }
    }
    var state = workgroupUniformLoad(&sharedRayState);

    loop {
        if (state.status != RAY_MARCHING || state.stepCount >= maxSteps) {
            break;
        }

        let point = localRayOrigin + state.depth * localRayDirection;
        let fieldValue = rbfField(point, lane);

        // Only the leader advances the ray; all lanes join the next evaluation.
        if (lane == 0u) {
            let distanceToSurface = abs(fieldValue);
            var hitSurface = fieldValue < 0.0;
            var nextDepth: f32;

            if (u32(sceneUniforms.stepStrategy) == STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ) {
                let lipschitzSteps = cellLocalLipschitzSteps(
                    distanceToSurface,
                    point,
                    localRayOrigin,
                    localRayDirection,
                    state.depth,
                );

                hitSurface = hitSurface || lipschitzSteps.surface <= epsilon;
                nextDepth = min(
                    state.depth + adjustedStep(lipschitzSteps.surface),
                    lipschitzSteps.cellExitDepth,
                );
            } else {
                nextDepth = state.depth + calculateStep(distanceToSurface, point);
            }

            sharedRayState.stepCount += 1u;
            if (hitSurface) {
                sharedRayState.distance = sharedRayState.depth;
                sharedRayState.status = RAY_HIT;
            } else {
                sharedRayState.depth = nextDepth;
                if (sharedRayState.depth >= endDepth || sharedRayState.stepCount >= maxSteps) {
                    sharedRayState.status = RAY_MISS;
                }
            }
        }
        state = workgroupUniformLoad(&sharedRayState);
    }

    return MarchingResult(state.distance, state.stepCount);
}

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

    let eye = sceneUniforms.cameraPosition.xyz;
    let ambient = 0.2 * baseColor;
    let diffuseColor = 0.6 * baseColor;
    let specularColor = vec3f(0.45, 0.45, 0.45);
    let shininess = 5.0;

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

@compute @workgroup_size(RAY_LANES, 1, 1)
fn main(
    @builtin(workgroup_id) groupId: vec3u,
    @builtin(local_invocation_index) lane: u32,
) {
    let screenSize = vec2i(textureDimensions(screenTexture));
    let screenPos = vec2i(groupId.xy);
    let pixelCenter = vec2f(screenPos) + vec2f(0.5);
    let uv = pixelCenter / vec2f(screenSize);
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let aspect = f32(screenSize.x) / f32(screenSize.y);
    let tanHalfFovY = tan(sceneUniforms.fieldOfView * 0.5);
    let rayOrigin = sceneUniforms.cameraPosition.xyz;
    let rayDirection = normalize(
        sceneUniforms.cameraForward.xyz +
        ndc.x * aspect * tanHalfFovY * sceneUniforms.cameraRight.xyz +
        ndc.y * tanHalfFovY * sceneUniforms.cameraUp.xyz
    );

    let result = shortestDistanceToSurface(rayOrigin, rayDirection, lane);
    if (lane != 0u) {
        return;
    }

    let pixelIndex = groupId.y * u32(screenSize.x) + groupId.x;
    let metricIndex = pixelIndex * 3u;
    metricsWorkBuffer[metricIndex] = result.distance;
    metricsWorkBuffer[metricIndex + 1u] = f32(result.stepCount);
    metricsWorkBuffer[metricIndex + 2u] = select(
        0.0, 1.0, result.distance < sceneUniforms.maxDistance,
    );

    if (result.distance >= sceneUniforms.maxDistance) {
        let a = 0.5 * rayDirection.y + 1.0;
        //let gradientColor = (1.0 - a)*vec4f(0.984, 0.894, 0.871, 1.0) + a*vec4f(0.820, 0.812, 0.847, 1.0);

        let topColor = vec3f(0.820, 0.812, 0.847);
        let middleColor = vec3f(0.914, 0.859, 0.859);
        let bottomColor = vec3f(0.984, 0.894, 0.871);

        let t = clamp(uv.y, 0.0, 1.0);

        var gradientColor: vec3f;
        if (t < 0.4) {
            gradientColor = mix(topColor, middleColor, t / 0.4);
        }

        gradientColor = mix(middleColor, bottomColor, (t - 0.4) / 0.6);
        textureStore(screenTexture, screenPos, vec4f(gradientColor, 1.0));
        return;
    }

    if (sceneUniforms.renderMode == 1.0) {
        let intensity = f32(result.stepCount) / sceneUniforms.maxSteps;
        let color = mix(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), intensity);
        textureStore(screenTexture, screenPos, vec4f(color, 1.0));
        return;
    }

    let point = rayOrigin + rayDirection * result.distance;
    let normal = normalize(rbfGradient(point));
    let shaded = phong(vec3f(0.96, 0.84, 0.23), point, normal);
    textureStore(screenTexture, screenPos, vec4f(shaded, 1.0));
}
