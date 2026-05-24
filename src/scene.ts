import { Camera } from "./camera";
import { Sphere } from "./sphere";

export class Scene {
    camera: Camera;
    spheres: Sphere[];

    constructor(numSpheres: number = 32) {
        this.spheres = new Array(numSpheres);
        for (let i = 0; i < this.spheres.length; i++) {
            const center: number[] = [
                -50.0 + 100.0 * Math.random(),
                -50.0 + 100.0 * Math.random(),
                -50.0 + 100.0 * Math.random(),
            ];

            const radius: number = 0.1 + 1.9 * Math.random();

            const color: number[] = [
                0.3 + 0.7 * Math.random(),
                0.3 + 0.7 * Math.random(),
                0.3 + 0.7 * Math.random(),
            ];

            this.spheres[i] = new Sphere(center, radius, color);
        }

        this.camera = new Camera([-20.0, 0.0, 0.0]);
    }
}
