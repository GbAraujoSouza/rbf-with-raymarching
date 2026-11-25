import { Mat4, Vec3 } from "wgpu-matrix";
import { RbfKernel } from "./rbf";
import { RenderMode, StepStrategy } from "./experiment";

export interface SceneUniformInput {
    screenWidth: number;
    screenHeight: number;
    sampleCount: number;
    debugPoints: number;

    cameraPosition: Vec3;
    cameraForward: Vec3;
    cameraRight: Vec3;
    cameraUp: Vec3;

    correctionPower: number;
    correctionLinear: number;
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
    fieldOfView: number;
    lipschitzGridDimensions: Vec3;
}

export interface LocalCellUniformInput {
    boxMin: Vec3;
    boxMax: Vec3;
    resolution: number;
    samplesPerAxis: number;
    kernelType: number;
    centerCount: number;
    gaussianEpsilon: number;
}

export abstract class UniformInput {
    declare data: Float32Array<ArrayBuffer>;
    declare uniformFloats: number;
    declare uniformBytes: number;

    declare uniformSlots: Record<string, number>;

    abstract createUniformData(input: any): Float32Array<ArrayBuffer>;

    protected setVec4(
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

    protected setVec3WithPadding(
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
}

export class LocalCellUniform extends UniformInput {
    declare data: Float32Array<ArrayBuffer>;

    override uniformFloats = 16;
    override uniformBytes = this.uniformFloats * 4;

    override uniformSlots = {
        boxMin: 0,
        boxMax: 4,
        resolution: 8,
        samplesPerAxis: 9,
        kernelType: 10,
        centerCount: 11,
        gaussianEpsilon: 12,
    };

    constructor(input: LocalCellUniformInput) {
        super();
        this.data = this.createUniformData(input);
    }

    createUniformData(input: LocalCellUniformInput): Float32Array<ArrayBuffer> {
        const uniformData = new Float32Array(this.uniformFloats);

        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.boxMin,
            input.boxMin[0],
            input.boxMin[1],
            input.boxMin[2],
        );

        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.boxMax,
            input.boxMax[0],
            input.boxMax[1],
            input.boxMax[2],
        );

        this.setVec4(
            uniformData,
            this.uniformSlots.resolution,
            input.resolution,
            input.samplesPerAxis,
            input.kernelType,
            input.centerCount,
        );

        this.setVec4(
            uniformData,
            this.uniformSlots.gaussianEpsilon,
            input.gaussianEpsilon,
            0.0,
            0.0,
            0.0,
        );

        return uniformData;
    }
}

export class SceneUniforms extends UniformInput {
    declare data: Float32Array<ArrayBuffer>;

    override uniformFloats = 64;
    override uniformBytes = this.uniformFloats * 4;

    override uniformSlots = {
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
        lipschitzGridDimensions: 60,
    };

    constructor(input: SceneUniformInput) {
        super();
        this.data = this.createUniformData(input);
    }

    override createUniformData(
        input: SceneUniformInput,
    ): Float32Array<ArrayBuffer> {
        const uniformData = new Float32Array(this.uniformFloats);

        // scene and counts
        this.setVec4(
            uniformData,
            this.uniformSlots.screenAndCounts,
            input.screenWidth,
            input.screenHeight,
            input.sampleCount,
            input.debugPoints,
        );

        // camera vectors
        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.cameraPosition,
            input.cameraPosition[0],
            input.cameraPosition[1],
            input.cameraPosition[2],
        );
        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.cameraForward,
            input.cameraForward[0],
            input.cameraForward[1],
            input.cameraForward[2],
        );
        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.cameraRight,
            input.cameraRight[0],
            input.cameraRight[1],
            input.cameraRight[2],
        );
        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.cameraUp,
            input.cameraUp[0],
            input.cameraUp[1],
            input.cameraUp[2],
        );

        // march params
        this.setVec4(
            uniformData,
            this.uniformSlots.marchParams,
            input.correctionPower,
            input.correctionLinear,
            input.maxDistance,
            input.epsilon,
        );

        // rbf params
        this.setVec4(
            uniformData,
            this.uniformSlots.rbfParams,
            input.gaussianEpsilon,
            input.debugPointRadius,
            input.stepStrategy,
            input.renderMode,
        );

        // light position
        this.setVec4(
            uniformData,
            this.uniformSlots.lightAndKernel,
            input.lightPosition[0],
            input.lightPosition[1],
            input.lightPosition[2],
            input.kernelType,
        );

        // bounding box points
        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.boxMin,
            input.boxMin[0],
            input.boxMin[1],
            input.boxMin[2],
        );
        this.setVec3WithPadding(
            uniformData,
            this.uniformSlots.boxMax,
            input.boxMax[0],
            input.boxMax[1],
            input.boxMax[2],
        );

        uniformData.set(input.worldToObject, this.uniformSlots.worldToObject);

        this.setVec4(
            uniformData,
            this.uniformSlots.maxSteps,
            input.maxSteps,
            input.fieldOfView,
            0.0,
            0.0,
        );
        this.setVec4(
            uniformData,
            this.uniformSlots.lipschitzGridDimensions,
            input.lipschitzGridDimensions[0],
            input.lipschitzGridDimensions[1],
            input.lipschitzGridDimensions[2],
            0.0,
        );
        return uniformData;
    }
}
