import { RbfKernel, type RbfFitConfig } from "./rbf";

export enum StepStrategy {
    "naive" = 0,
    "exponential_correction" = 1,
    "gradient" = 2,
}

export enum RenderMode {
    "shaded" = 0,
    "heatmap" = 1,
}

export interface RayMarchingConfig {
    strategy: StepStrategy;
    epsilon: number;
    maxDistance: number;
    maxSteps: number;
    correctionLinear: number;
    correctionPower: number;
}

export interface ExperimentRbfConfig extends RbfFitConfig {
    debugPointRadius: number;
}

export type SceneId = "sphere" | "torus" | "dragon";

export interface ExperimentState {
    sceneId: SceneId;
    rayMarchingConfig: RayMarchingConfig;
    rbfConfig: ExperimentRbfConfig;
    renderMode: RenderMode;
    showControlPoints: boolean;
}

export const DEFAULT_EXPERIMENT_STATE: ExperimentState = {
    sceneId: "sphere",
    renderMode: RenderMode.shaded,
    rayMarchingConfig: {
        strategy: StepStrategy.naive,
        epsilon: 1e-5,
        maxDistance: 20,
        maxSteps: 255,
        correctionLinear: 0.9,
        correctionPower: 0.85,
    },
    rbfConfig: {
        surfaceSampleCount: 8,
        gaussianEpsilon: 1.35,
        sphereRadius: 0.7,
        debugPointRadius: 0.02,
        normalOffset: 0.1,
        regularization: 0.001,
        kernel: RbfKernel.linear,
    },
    showControlPoints: false,
};
