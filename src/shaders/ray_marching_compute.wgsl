struct SceneUniforms {
    screenAndCounts: vec4f, // width, height, # points, show_debug 
    cameraPosition: vec4f,
    cameraForward: vec4f,
    cameraRight: vec4f,
    cameraUp: vec4f,
    
    // RAY MARCHING PARAMS
    gaussianKernelCorrectionPower: f32,
    gaussianKernelCorrectionLinear: f32,
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

    maxSteps: f32, //this has padding
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
    steps: u32,
}

struct CellLocalLipschitzSteps {
    surface: f32,
    cellExit: f32,
}

const STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ: u32 = 3u;
const MINIMUM_LIPSCHITZ_BOUND: f32 = 1e-4;
const GRID_BOUNDARY_TOLERANCE: f32 = 1e-5;
const DIRECTION_EPSILON: f32 = 1e-8;
const LARGE_DISTANCE: f32 = 1e30;


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
            let radius = length(point - center);
            if (radius == 0.0) {
                return 0.0;
            }
            return radius * radius * log(radius);
        }
        default {
            // default to lenear
            return length(point - center);
        }
    }
}

fn rbfField(point: vec3f) -> f32 {
    let sampleCount = i32(sceneUniforms.screenAndCounts.z);
    var total = 0.0;

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let weight = sampleWeights.values[index];
        total += kernel(point, center) * weight;
    }

    return total;
}

fn pointsSdf(point: vec3f) -> f32 {
    let sampleCount = i32(sceneUniforms.screenAndCounts.z);
    var minimumDistance = sceneUniforms.maxDistance;

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let distanceToPoint = distance(point, center) - sceneUniforms.debugPointRadius;
        minimumDistance = min(minimumDistance, distanceToPoint);
    }

    return minimumDistance;
}

fn sceneSdf(point: vec3f, usePoints: bool) -> f32 {
    if (usePoints) {
        return pointsSdf(point);
    }

    return rbfField(point);
    // let rbfSurfaceDist = rbfField(point);
    // let groundDist = point.y - 0.1;
    // return min(rbfSurfaceDist, groundDist);
}

fn calculateStep(distanceToSurface: f32, point: vec3f) -> f32 {
    var strategyId: u32 = u32(sceneUniforms.stepStrategy);
    switch(strategyId) {
        case 0u {
            // naive
            return distanceToSurface;
        }
        case 1u {
            // exponential correction
            let epsilon = sceneUniforms.epsilon;
            let correctionPower = sceneUniforms.gaussianKernelCorrectionPower;
            let correctionLinear = sceneUniforms.gaussianKernelCorrectionLinear;
            return correctionLinear * pow(max(distanceToSurface, epsilon), correctionPower);
        }
        case 2u {
            // gradient
            var gradient: vec3f = estimateNormal(point, false, false, true);
            let rawStep = distanceToSurface / max(length(gradient), 0.01);
            return rawStep;
        }
        default {
            return distanceToSurface;
        }
    }
}

fn lipschitzGridDimensions() -> vec3u {
    return vec3u(sceneUniforms.lipschitzGridDimensions.xyz);
}

fn lipschitzCellCoordinate(
    normalizedPosition: f32,
    dimension: u32,
    rayDirection: f32,
) -> u32 {
    let scaledPosition =
        clamp(normalizedPosition, 0.0, 1.0) * f32(dimension);
    let nearestBoundary = round(scaledPosition);
    var coordinate = i32(floor(scaledPosition));
    if (
        abs(scaledPosition - nearestBoundary) <= GRID_BOUNDARY_TOLERANCE
    ) {
        coordinate = i32(nearestBoundary);
        if (rayDirection < 0.0) {
            coordinate -= 1;
        }
    }
    return u32(clamp(coordinate, 0, i32(dimension) - 1));
}

fn lipschitzCellForPoint(point: vec3f, rayDirection: vec3f) -> vec3u {
    let dimensions = lipschitzGridDimensions();
    let boxMin = sceneUniforms.boxMin.xyz;
    let boxMax = sceneUniforms.boxMax.xyz;
    let normalizedPosition =
        (point - boxMin) / (boxMax - boxMin);
    return vec3u(
        lipschitzCellCoordinate(
            normalizedPosition.x,
            dimensions.x,
            rayDirection.x,
        ),
        lipschitzCellCoordinate(
            normalizedPosition.y,
            dimensions.y,
            rayDirection.y,
        ),
        lipschitzCellCoordinate(
            normalizedPosition.z,
            dimensions.z,
            rayDirection.z,
        ),
    );
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

fn distanceToCellExit(
    point: vec3f,
    rayDirection: vec3f,
    cell: vec3u,
) -> f32 {
    let dimensions = lipschitzGridDimensions();
    let boxMin = sceneUniforms.boxMin.xyz;
    let boxMax = sceneUniforms.boxMax.xyz;
    let cellSize = (boxMax - boxMin) / vec3f(dimensions);
    let cellMin = boxMin + vec3f(cell) * cellSize;
    let cellMax = cellMin + cellSize;

    var xExit = LARGE_DISTANCE;
    if (abs(rayDirection.x) > DIRECTION_EPSILON) {
        let xBoundary = select(cellMin.x, cellMax.x, rayDirection.x > 0.0);
        xExit = max((xBoundary - point.x) / rayDirection.x, 0.0);
    }

    var yExit = LARGE_DISTANCE;
    if (abs(rayDirection.y) > DIRECTION_EPSILON) {
        let yBoundary = select(cellMin.y, cellMax.y, rayDirection.y > 0.0);
        yExit = max((yBoundary - point.y) / rayDirection.y, 0.0);
    }

    var zExit = LARGE_DISTANCE;
    if (abs(rayDirection.z) > DIRECTION_EPSILON) {
        let zBoundary = select(cellMin.z, cellMax.z, rayDirection.z > 0.0);
        zExit = max((zBoundary - point.z) / rayDirection.z, 0.0);
    }

    return min(xExit, min(yExit, zExit));
}

fn cellLocalLipschitzSteps(
    distanceToSurface: f32,
    point: vec3f,
    rayDirection: vec3f,
) -> CellLocalLipschitzSteps {
    let cell = lipschitzCellForPoint(point, rayDirection);
    let directionScale = max(length(rayDirection), DIRECTION_EPSILON);
    let surfaceStep =
        distanceToSurface /
        (localLipschitzBound(cell) * directionScale);
    let cellExitStep = distanceToCellExit(point, rayDirection, cell);
    return CellLocalLipschitzSteps(surfaceStep, cellExitStep);
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

/*
    Ray march function
*/
fn shortestDistanceToSurface(rayOrigin: vec3f, rayDirection: vec3f, usePoints: bool) -> MarchingResult {
    let maxDistance = sceneUniforms.maxDistance;
    let epsilon = sceneUniforms.epsilon;
    let boxMin = sceneUniforms.boxMin.xyz;
    let boxMax = sceneUniforms.boxMax.xyz;

    let localRayOrigin: vec3f = (sceneUniforms.worldToObject * vec4(rayOrigin, 1.0)).xyz;
    let localRayDirection: vec3f = (sceneUniforms.worldToObject * vec4(rayDirection, 0.0)).xyz;

    var result: MarchingResult;

    let hitBox: RayBoxIntercept = intersectAABB(localRayOrigin, localRayDirection, boxMin, boxMax);
    if (!hitBox.hit) {
        result.distance = maxDistance;
        result.steps = 0;
        return result;
    }

    var depth = max(hitBox.tMin, 0.0);
    var endDepth = min(hitBox.tMax, maxDistance);
    for (var step: u32 = 0; step < u32(sceneUniforms.maxSteps); step += 1) {
        let point = localRayOrigin + depth * localRayDirection;
        let fieldValue = sceneSdf(point, usePoints);
        let distanceToSurface = abs(fieldValue);
        let strategyId = u32(sceneUniforms.stepStrategy);
        var lipschitzSteps = CellLocalLipschitzSteps(
            distanceToSurface,
            LARGE_DISTANCE,
        );
        if (
            strategyId == STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ &&
            !usePoints
        ) {
            lipschitzSteps = cellLocalLipschitzSteps(
                distanceToSurface,
                point,
                localRayDirection,
            );
        }

        if (usePoints) {
            if (distanceToSurface < epsilon) {
                result.distance = depth;
                result.steps = step;
                return result;
            }
        } else {
            if (fieldValue < 0.0) {
                result.distance = depth;
                result.steps = step;
                return result;
            }
            if (strategyId == STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ) {
                if (lipschitzSteps.surface <= epsilon) {
                    result.distance = depth;
                    result.steps = step;
                    return result;
                }
            }
        }

        var stepDistance = distanceToSurface;
        if (!usePoints) {
            if (strategyId == STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ) {
                stepDistance = min(
                    lipschitzSteps.surface,
                    lipschitzSteps.cellExit,
                );
            } else {
                // make the correction to the step
                stepDistance = calculateStep(distanceToSurface, point);
            }
            
        }

        if (
            strategyId == STEP_STRATEGY_CELL_LOCAL_LIPSCHITZ &&
            !usePoints
        ) {
            depth += stepDistance;
        } else {
            depth += max(stepDistance, epsilon);
        }

        if (depth >= endDepth) {
            result.distance = maxDistance;
            result.steps = step;
            return result;
        }
    }

    result.distance = maxDistance;
    result.steps = u32(sceneUniforms.maxSteps);
    return result;
}

fn rbfGradient(point: vec3f) -> vec3f {
    let sampleCount = i32(sceneUniforms.screenAndCounts.z);
    var gradient = vec3f(0.0, 0.0, 0.0);

    for (var index = 0; index < sampleCount; index += 1) {
        let center = samplePositions.values[index].xyz;
        let offset = point - center;
        let radius = length(offset);
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
                    derivativeScale = 2.0 * log(radius) + 1.0;
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

fn estimateNormal(
    point: vec3f,
    usePoints: bool,
    normal: bool,
    analyticalGradient: bool,
) -> vec3f {
    let epsilon = sceneUniforms.epsilon;
    var gradient: vec3f;
    if (analyticalGradient && !usePoints) {
        gradient = rbfGradient(point);
    } else {
        let e = vec2f(epsilon, 0.0);
        gradient = (
            sceneSdf(point, usePoints) -
            vec3f(
                sceneSdf(point - e.xyy, usePoints),
                sceneSdf(point - e.yxy, usePoints),
                sceneSdf(point - e.yyx, usePoints),
            )
        ) / epsilon;
    }
    // let xOffset = vec3f(epsilon, 0.0, 0.0);
    // let yOffset = vec3f(0.0, epsilon, 0.0);
    // let zOffset = vec3f(0.0, 0.0, epsilon);

    // let gradient = vec3f(
    //     sceneSdf(point + xOffset, usePoints) - sceneSdf(point - xOffset, usePoints),
    //     sceneSdf(point + yOffset, usePoints) - sceneSdf(point - yOffset, usePoints),
    //     sceneSdf(point + zOffset, usePoints) - sceneSdf(point - zOffset, usePoints),
    // );

    if (normal) {
        return normalize(gradient);
    }
    return gradient;
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

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationId: vec3u) {

    let screenSize: vec2i = vec2i(textureDimensions(screenTexture));
    let screenPos: vec2i = vec2i(i32(GlobalInvocationId.x), i32(GlobalInvocationId.y));
    if (screenPos.x >= screenSize.x || screenPos.y >= screenSize.y) {
        return;
    }

    let horizontalCoeff: f32 = (f32(screenPos.x) - f32(screenSize.x) / 2.0) / f32(screenSize.x);
    //let verticalCoeff: f32 = (f32(screenPos.y) - f32(screenSize.y) / 2.0) / f32(screenSize.x);
    let verticalCoeff: f32 = -(f32(screenPos.y) - f32(screenSize.y) / 2.0) / f32(screenSize.x);


    let rayOrigin = sceneUniforms.cameraPosition.xyz;
    let rayDirection = normalize(sceneUniforms.cameraForward.xyz + horizontalCoeff * sceneUniforms.cameraRight.xyz + verticalCoeff * sceneUniforms.cameraUp.xyz);
    //let rayDir = rayDirection(input.uv);
    let showPoints = sceneUniforms.screenAndCounts.w > 0.5;

    // Call "RayMarch" function
    let rbfResult: MarchingResult = shortestDistanceToSurface(rayOrigin, rayDirection, false);

    metricsWorkBuffer[
        GlobalInvocationId.y * 
        u32(screenSize.x) + 
        GlobalInvocationId.x] = f32(rbfResult.distance);

    let rbfDistance = rbfResult.distance;

    var hitDistance = rbfDistance;
    var hitPoints = false;

    if (showPoints) {
        let pointsResult = shortestDistanceToSurface(rayOrigin, rayDirection, true);
        let pointsDistance = pointsResult.distance;
        if (pointsDistance < hitDistance) {
            hitDistance = pointsDistance;
            hitPoints = true;
        }
    }

    if (hitDistance >= sceneUniforms.maxDistance) {
        textureStore(screenTexture, screenPos, vec4f(0.8118, 0.9333, 1.0, 1.0));
        return;
    }

    let point = rayOrigin + rayDirection * hitDistance;
    let normal = estimateNormal(point, hitPoints, true, !hitPoints);
    let baseColor = select(vec3f(0.93, 0.32, 0.25), vec3f(0.96, 0.84, 0.23), hitPoints);

    // final color with light
    let shaded = phong(baseColor, point, normal);

    if (sceneUniforms.renderMode == 1) {
        let steps = rbfResult.steps;
        let stepIntensity = f32(steps)/f32(u32(sceneUniforms.maxSteps));
        // Heatmap: Blue (cold/few steps) to Red (hot/many steps)
        let stepGradientColor = mix(vec3f(0.0, 0.0, 1.0), vec3f(1.0, 0.0, 0.0), stepIntensity);

        textureStore(screenTexture, screenPos, vec4f(stepGradientColor, 1.0));
        return;
    }

    //return vec4f(shaded, 1.0);
    textureStore(screenTexture, screenPos, vec4f(shaded, 1.0));
}
