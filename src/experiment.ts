import type { RbfFitConfig } from "./rbf";

export type StepStrategy = "naive" | "exponential_correction" | "gradient";

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

export interface ExperimentState {
    sceneId: string;
    rayMarchingConfig: RayMarchingConfig;
    rbfConfig: ExperimentRbfConfig;
    renderMode: RenderMode;
    showControlPoints: boolean;
}

export const DEFAULT_EXPERIMENT_STATE: ExperimentState = {
    sceneId: "123",
    renderMode: RenderMode.shaded,
    rayMarchingConfig: {
        strategy: "exponential_correction",
        epsilon: 1e-5,
        maxDistance: 20,
        maxSteps: 255,
        correctionLinear: 0.9,
        correctionPower: 0.85,
    },
    rbfConfig: {
        surfaceSampleCount: 32,
        gaussianEpsilon: 1.35,
        sphereRadius: 1,
        debugPointRadius: 0.06,
    },
    showControlPoints: true,
};
