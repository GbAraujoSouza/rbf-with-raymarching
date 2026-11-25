import { RbfKernel, type RbfFitConfig } from "./rbf";

export enum StepStrategy {
    "naive" = 0,
    "gradient" = 1,
    "cellLocalLipschitz" = 2,
    "stepTracing" = 3,
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
        resolution: 16,
        isoValue: 0,
        extraPadding: 0,
    },
    rayMarchingConfig: {
        strategy: StepStrategy.naive,
        epsilon: 1e-4,
        maxDistance: 20,
        maxSteps: 255,
        correctionLinear: 1,
        correctionPower: 1,
        lipschitzGridResolution: 8,
        lipschitzSamplesPerAxis: 3,
    },
    rbfConfig: {
        surfaceSampleCount: 1000,
        gaussianEpsilon: 1.35,
        sphereRadius: 1.5,
        debugPointRadius: 0.02,
        normalOffset: 0.03,
        regularization: 0.0,
        kernel: RbfKernel.thinPlate,
    },
    showControlPoints: false,
    renderBackend: RenderBackend.rayMarching,
};
