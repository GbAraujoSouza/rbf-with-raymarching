import { RbfKernel, type RbfFitConfig } from "./rbf";

export enum StepStrategy {
    "naive" = 0,
    "exponentialCorrection" = 1,
    "gradient" = 2,
    "cellLocalLipschitz" = 3,
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
    lipschitzGridResolution: number;
    lipschitzSamplesPerAxis: number;
    lipschitzSafetyFactor: number;
}

export interface MarchingCubesConfig {
    resolution: number;
    isoValue: number;
    extraPadding: number;
}

export interface ExperimentRbfConfig extends RbfFitConfig {
    debugPointRadius: number;
}

export type SceneId =
    | "sphere"
    | "torus"
    | "dragon"
    | "bunny"
    | "buddha"
    | "teapot";

export enum RenderBackend {
    rayMarching = 0,
    marchingCubes = 1,
}

export interface ExperimentState {
    sceneId: SceneId;
    rayMarchingConfig: RayMarchingConfig;
    marchingCubesConfig: MarchingCubesConfig;
    rbfConfig: ExperimentRbfConfig;
    renderMode: RenderMode;
    showControlPoints: boolean;
    renderBackend: RenderBackend;
}

export const DEFAULT_EXPERIMENT_STATE: ExperimentState = {
    sceneId: "bunny",
    renderMode: RenderMode.shaded,
    marchingCubesConfig: {
        resolution: 64,
        isoValue: 0,
        extraPadding: 0,
    },
    rayMarchingConfig: {
        strategy: StepStrategy.exponentialCorrection,
        epsilon: 1e-4,
        maxDistance: 20,
        maxSteps: 255,
        correctionLinear: 0.9,
        correctionPower: 0.85,
        lipschitzGridResolution: 8,
        lipschitzSamplesPerAxis: 3,
        lipschitzSafetyFactor: 1.2,
    },
    rbfConfig: {
        surfaceSampleCount: 200,
        gaussianEpsilon: 1.35,
        sphereRadius: 1,
        debugPointRadius: 0.02,
        normalOffset: 0.1,
        regularization: 0.001,
        kernel: RbfKernel.linear,
    },
    showControlPoints: false,
    renderBackend: RenderBackend.rayMarching,
};
