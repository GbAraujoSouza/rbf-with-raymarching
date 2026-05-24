import { glMatrix, vec3 } from "gl-matrix";

export class Camera {
    position: Float32Array;
    theta: number;
    phi: number;
    forward!: Float32Array;
    up!: Float32Array;
    right!: Float32Array;

    constructor(position: number[]) {
        this.position = new Float32Array(position);
        this.theta = 0.0;
        this.phi = 0.0;

        this.recalculateVectors();
    }

    recalculateVectors() {
        const thetaRad = glMatrix.toRadian(this.theta);
        const phiRad = glMatrix.toRadian(this.phi);

        this.forward = new Float32Array([
            Math.cos(thetaRad) * Math.cos(phiRad),
            Math.sin(thetaRad) * Math.cos(phiRad),
            Math.sin(phiRad),
        ]);

        this.right = new Float32Array([0.0, 0.0, 0.0]);
        vec3.cross(this.right, this.forward, [0.0, 0.0, 1.0]);

        this.up = new Float32Array([0.0, 0.0, 0.0]);
        vec3.cross(this.up, this.right, this.forward);
    }
}
