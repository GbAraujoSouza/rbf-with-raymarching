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

export function evaluateKernelDerivativeScale(
    radius: number,
    kernel: RbfKernel,
    epsilon: number,
): number {
    switch (kernel) {
        case RbfKernel.linear:
            return radius > 0 ? 1 / radius : 0;
        case RbfKernel.gaussian: {
            const epsilonSquared = epsilon * epsilon;
            return (
                -2 *
                epsilonSquared *
                Math.exp(-epsilonSquared * radius * radius)
            );
        }
        case RbfKernel.cubic:
            return 3 * radius;
        case RbfKernel.quintic:
            return 5 * radius * radius * radius;
        case RbfKernel.thinPlate:
            return radius > 0 ? 2 * Math.log(radius) + 1 : 0;
        default:
            return radius > 0 ? 1 / radius : 0;
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
