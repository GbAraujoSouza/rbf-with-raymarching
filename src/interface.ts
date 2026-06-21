import { Renderer } from "./renderer";
import { StepStrategy } from "./experiment";

export function setupUI(renderer: Renderer) {
    const experimentState = renderer.experimentState;

    // Show Control Points
    const chkControlPoints = document.getElementById(
        "chk-control-points",
    ) as HTMLInputElement;
    if (chkControlPoints) {
        chkControlPoints.checked = experimentState.showControlPoints;
        chkControlPoints.addEventListener("change", (e) => {
            experimentState.showControlPoints = (
                e.target as HTMLInputElement
            ).checked;
        });
    }

    const chkHeatMap = document.getElementById(
        "chk-heat-map",
    ) as HTMLInputElement;
    if (chkHeatMap) {
        chkHeatMap.checked = !!experimentState.renderMode;
        chkHeatMap.addEventListener("change", (e) => {
            experimentState.renderMode = (e.target as HTMLInputElement).checked
                ? 1
                : 0;
        });
    }

    // Ray Marching Strategy
    const selStrategy = document.getElementById(
        "sel-strategy",
    ) as HTMLSelectElement;
    if (selStrategy) {
        selStrategy.value =
            StepStrategy[experimentState.rayMarchingConfig.strategy];
        selStrategy.addEventListener("change", (e) => {
            experimentState.rayMarchingConfig.strategy =
                StepStrategy[
                    (e.target as HTMLSelectElement)
                        .value as keyof typeof StepStrategy
                ];
        });
    }

    // Bind slider helper
    function bindSlider(
        id: string,
        valId: string,
        obj: any,
        key: string,
        isInt: boolean = false,
    ) {
        const slider = document.getElementById(id) as HTMLInputElement;
        const valDisplay = document.getElementById(valId) as HTMLSpanElement;

        if (!slider || !valDisplay) return;

        slider.value = obj[key].toString();
        valDisplay.innerText = obj[key].toString();

        slider.addEventListener("input", (e) => {
            const valStr = (e.target as HTMLInputElement).value;
            const val = isInt ? parseInt(valStr, 10) : parseFloat(valStr);
            obj[key] = val;
            valDisplay.innerText = valStr;
        });
    }

    // Ray Marching params
    bindSlider(
        "rng-rm-epsilon",
        "val-rm-epsilon",
        experimentState.rayMarchingConfig,
        "epsilon",
    );
    bindSlider(
        "rng-rm-maxdist",
        "val-rm-maxdist",
        experimentState.rayMarchingConfig,
        "maxDistance",
    );
    bindSlider(
        "rng-rm-maxsteps",
        "val-rm-maxsteps",
        experimentState.rayMarchingConfig,
        "maxSteps",
        true,
    );
    bindSlider(
        "rng-rm-corlinear",
        "val-rm-corlinear",
        experimentState.rayMarchingConfig,
        "correctionLinear",
    );
    bindSlider(
        "rng-rm-corpower",
        "val-rm-corpower",
        experimentState.rayMarchingConfig,
        "correctionPower",
    );

    // RBF Config params
    bindSlider(
        "rng-rbf-samples",
        "val-rbf-samples",
        experimentState.rbfConfig,
        "surfaceSampleCount",
        true,
    );
    bindSlider(
        "rng-rbf-gEpsilon",
        "val-rbf-gEpsilon",
        experimentState.rbfConfig,
        "gaussianEpsilon",
    );
    bindSlider(
        "rng-rbf-radius",
        "val-rbf-radius",
        experimentState.rbfConfig,
        "sphereRadius",
    );
    bindSlider(
        "rng-rbf-debugRadius",
        "val-rbf-debugRadius",
        experimentState.rbfConfig,
        "debugPointRadius",
    );
    bindSlider(
        "rng-rbf-offset",
        "val-rbf-offset",
        experimentState.rbfConfig,
        "normalOffset",
    );
    bindSlider(
        "rng-rbf-regularization",
        "val-rbf-regularization",
        experimentState.rbfConfig,
        "regularization",
    );

    // Apply RBF config button
    const btnApplyRbf = document.getElementById(
        "btn-apply-rbf",
    ) as HTMLButtonElement;
    if (btnApplyRbf) {
        btnApplyRbf.addEventListener("click", () => {
            renderer.rebuildRbfAssets();
        });
    }
}
