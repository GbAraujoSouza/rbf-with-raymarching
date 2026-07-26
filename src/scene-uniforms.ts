import { Mat4, Vec3 } from "wgpu-matrix";
import { RbfKernel } from "./rbf";
import { RenderMode, StepStrategy } from "./experiment";

export const SCENE_UNIFORM_SLOTS = {
    screenAndCounts: 0,
    cameraPosition: 4,
    cameraForward: 8,
    cameraRight: 12,
    cameraUp: 16,
    marchParams: 20,
    rbfParams: 24,
    lightAndKernel: 28,
    boxMin: 32,
    boxMax: 36,
    worldToObject: 40,
    maxSteps: 56,
};

export const SCENE_UNIFORM_FLOATS = 60;
export const SCENE_UNIFORM_BYTES = SCENE_UNIFORM_FLOATS * 4;

export interface SceneUniformInput {
    screenWidth: number;
    screenHeight: number;
    sampleCount: number;
    debugPoints: number;

    cameraPosition: Vec3;
    cameraForward: Vec3;
    cameraRight: Vec3;
    cameraUp: Vec3;

    gaussianKernelCorrectionPower: number;
    gaussianKernelCorrectionLinear: number;
    maxDistance: number;
    epsilon: number;

    gaussianEpsilon: number;
    debugPointRadius: number;
    stepStrategy: StepStrategy;
    renderMode: RenderMode;

    lightPosition: Vec3;
    kernelType: RbfKernel;

    boxMin: Vec3;
    boxMax: Vec3;

    worldToObject: Mat4;

    maxSteps: number;
}

export class SceneUniforms {
    public static setVec4(
        data: Float32Array,
        offset: number,
        x: number,
        y: number,
        z: number,
        w: number,
    ) {
        data[offset] = x;
        data[offset + 1] = y;
        data[offset + 2] = z;
        data[offset + 3] = w;
    }

    public static setVec3WithPadding(
        data: Float32Array,
        offset: number,
        x: number,
        y: number,
        z: number,
    ) {
        data[offset] = x;
        data[offset + 1] = y;
        data[offset + 2] = z;
        data[offset + 3] = 0.0;
    }

    public static createSceneUniformData(
        input: SceneUniformInput,
    ): Float32Array<ArrayBuffer> {
        const uniformData = new Float32Array(SCENE_UNIFORM_FLOATS);

        // scene and counts
        SceneUniforms.setVec4(
            uniformData,
            SCENE_UNIFORM_SLOTS.screenAndCounts,
            input.screenWidth,
            input.screenHeight,
            input.sampleCount,
            input.debugPoints,
        );

        // camera vectors
        SceneUniforms.setVec3WithPadding(
            uniformData,
            SCENE_UNIFORM_SLOTS.cameraPosition,
            input.cameraPosition[0],
            input.cameraPosition[1],
            input.cameraPosition[2],
        );
        SceneUniforms.setVec3WithPadding(
            uniformData,
            SCENE_UNIFORM_SLOTS.cameraForward,
            input.cameraForward[0],
            input.cameraForward[1],
            input.cameraForward[2],
        );
        SceneUniforms.setVec3WithPadding(
            uniformData,
            SCENE_UNIFORM_SLOTS.cameraRight,
            input.cameraRight[0],
            input.cameraRight[1],
            input.cameraRight[2],
        );
        SceneUniforms.setVec3WithPadding(
            uniformData,
            SCENE_UNIFORM_SLOTS.cameraUp,
            input.cameraUp[0],
            input.cameraUp[1],
            input.cameraUp[2],
        );

        // march params
        SceneUniforms.setVec4(
            uniformData,
            SCENE_UNIFORM_SLOTS.marchParams,
            input.gaussianKernelCorrectionPower,
            input.gaussianKernelCorrectionLinear,
            input.maxDistance,
            input.epsilon,
        );

        // rbf params
        SceneUniforms.setVec4(
            uniformData,
            SCENE_UNIFORM_SLOTS.rbfParams,
            input.gaussianEpsilon,
            input.debugPointRadius,
            input.stepStrategy,
            input.renderMode,
        );

        // light position
        SceneUniforms.setVec4(
            uniformData,
            SCENE_UNIFORM_SLOTS.lightAndKernel,
            input.lightPosition[0],
            input.lightPosition[1],
            input.lightPosition[2],
            input.kernelType,
        );

        // bounding box points
        SceneUniforms.setVec3WithPadding(
            uniformData,
            SCENE_UNIFORM_SLOTS.boxMin,
            input.boxMin[0],
            input.boxMin[1],
            input.boxMin[2],
        );
        SceneUniforms.setVec3WithPadding(
            uniformData,
            SCENE_UNIFORM_SLOTS.boxMax,
            input.boxMax[0],
            input.boxMax[1],
            input.boxMax[2],
        );

        uniformData.set(input.worldToObject, SCENE_UNIFORM_SLOTS.worldToObject);

        SceneUniforms.setVec4(
            uniformData,
            SCENE_UNIFORM_SLOTS.maxSteps,
            input.maxSteps,
            0.0,
            0.0,
            0.0
        )
        return uniformData;
    }
}
