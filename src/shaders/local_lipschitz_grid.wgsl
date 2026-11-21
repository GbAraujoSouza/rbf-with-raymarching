struct GridUniforms {
    boxMin: vec4f,
    boxMax: vec4f,

    resolution: f32,
    samplesPerAxis: f32,
    kernelType: f32,
    centerCount: f32,

    gaussianEpsilon: f32,
}

@group(0) @binding(0)
var<uniform> gridUniforms: GridUniforms;

@group(0) @binding(1)
var<storage, read> positions: array<vec4f>;

@group(0) @binding(2)
var<storage, read> weights: array<f32>;

@group(0) @binding(3)
var<storage, read_write> values: array<f32>;


@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) cell: vec3u) {
    // Each invocation represents a uniform grid cell

    let resolution = u32(gridUniforms.resolution);
    let samplesPerAxis = u32(gridUniforms.samplesPerAxis);
    let boxMin = gridUniforms.boxMin.xyz;
    let boxMax = gridUniforms.boxMax.xyz;
    let cellSize = (boxMax - boxMin) / f32(resolution);

    var maxGradient = 0.0;
    for (var sampleZ = 0u; sampleZ < samplesPerAxis; sampleZ += 1) {
        for (var sampleY = 0u; sampleY < samplesPerAxis; sampleY += 1) {
            for (var sampleX = 0u; sampleX < samplesPerAxis; sampleX += 1) {
                // this is the size between 2 points in a cell, on every axis
                let ratio = vec3f(f32(sampleX), f32(sampleY), f32(sampleZ)) / f32(samplesPerAxis - 1);

                let pointPosition = vec3f(
                    boxMin + (vec3f(cell) + ratio) * cellSize 
                );

                let gradient: vec3f = rbfGradient(pointPosition);
                maxGradient = max(maxGradient, length(gradient));
            }
        }
    }

    let cellIndex = cell.x +
                    cell.y * resolution +
                    cell.z * resolution * resolution;

    values[cellIndex] = maxGradient;
}

fn rbfGradient(point: vec3f) -> vec3f {
    let sampleCount = u32(gridUniforms.centerCount);
    var gradient = vec3f(0.0, 0.0, 0.0);

    for (var index = 0u; index < sampleCount; index += 1) {
        let center: vec3f = positions[index].xyz;
        let offset: vec3f = point - center;
        let radius: f32 = length(offset);
        var derivativeScale: f32 = 0.0;

        switch(u32(gridUniforms.kernelType)) {
            case 0u {
                if (radius > 0.0) {
                    derivativeScale = 1.0 / radius;
                }
            }
            case 1u {
                let epsilon = gridUniforms.gaussianEpsilon;
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

        gradient += weights[index] * derivativeScale * offset;
    }

    return gradient;
}