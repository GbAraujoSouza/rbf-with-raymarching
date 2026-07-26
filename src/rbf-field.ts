import { Vec3, vec3 } from "wgpu-matrix";
import { RbfFitConfig, RbfKernel, RbfFitResult } from "./rbf";

export function evaluateKernel(
    point: Vec3,
    center: Vec3,
    kernel: RbfKernel,
    epsilon: number,
): number {
    const radius = vec3.distance(point, center);

    switch (kernel) {
        case RbfKernel.linear:
            return radius;
        case RbfKernel.gaussian:
            return Math.exp(-(epsilon * epsilon * radius * radius));
        case RbfKernel.cubic:
            return radius * radius * radius;
        case RbfKernel.quintic: {
            const radiusSquared = radius * radius;
            return radiusSquared * radiusSquared * radius;
        }
        case RbfKernel.thinPlate:
            if (radius === 0.0) {
                return 0.0;
            }
            return radius * radius * Math.log(radius);
        default:
            return radius;
    }
}

export function evaluateRbfField(
    point: Vec3,
    rbfFitResult: RbfFitResult,
    rbfFitConfig: RbfFitConfig,
): number {
    const totalSamples: number = rbfFitResult.samples.length;
    let total: number = 0;
    for (let i = 0; i < totalSamples; i++) {
        const center = rbfFitResult.samples[i].position;
        const weight = rbfFitResult.weights[i];
        total +=
            weight *
            evaluateKernel(
                point,
                center,
                rbfFitConfig.kernel,
                rbfFitConfig.gaussianEpsilon,
            );
    }

    return total;
}
