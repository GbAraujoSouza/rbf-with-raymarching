import { Renderer } from "./renderer";
import { RenderBackend, StepStrategy } from "./experiment";
import { RbfKernel } from "./rbf";

export function setupUI(
    renderer: Renderer,
    onRayMarchingConfigChanged?: () => void,
) {
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

    const selRenderBackend = document.getElementById(
        "sel-render-backend",
    ) as HTMLSelectElement;
    if (selRenderBackend) {
        selRenderBackend.value =
            RenderBackend[experimentState.renderBackend];
        selRenderBackend.addEventListener("change", (e) => {
            experimentState.renderBackend =
                RenderBackend[
                    (e.target as HTMLSelectElement)
                        .value as keyof typeof RenderBackend
                ];
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
            if (
                experimentState.rayMarchingConfig.strategy ===
                StepStrategy.cellLocalLipschitz
            ) {
                renderer.ensureLipschitzGrid();
            }
            onRayMarchingConfigChanged?.();
        });
    }

    const selKernel = document.getElementById(
        "sel-kernel",
    ) as HTMLSelectElement;
    if (selKernel) {
        selKernel.value = "linear";
        selKernel.addEventListener("change", (e) => {
            experimentState.rbfConfig.kernel =
                RbfKernel[
                    (e.target as HTMLSelectElement)
                        .value as keyof typeof RbfKernel
                ];
        });
    }

    const selScene = document.getElementById("sel-scene") as HTMLSelectElement;
    if (selScene) {
        selScene.value = experimentState.sceneId;
        selScene.addEventListener("change", (e) => {
            experimentState.sceneId = (e.target as HTMLSelectElement).value as any;
        });
    }

    // Bind slider helper
    function bindSlider(
        id: string,
        valId: string,
        obj: any,
        key: string,
        isInt: boolean = false,
        onChange?: () => void,
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
            onChange?.();
        });
    }

    // Ray Marching params
    bindSlider(
        "rng-rm-epsilon",
        "val-rm-epsilon",
        experimentState.rayMarchingConfig,
        "epsilon",
        false,
        onRayMarchingConfigChanged,
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
        false,
        onRayMarchingConfigChanged,
    );
    bindSlider(
        "rng-rm-corpower",
        "val-rm-corpower",
        experimentState.rayMarchingConfig,
        "correctionPower",
        false,
        onRayMarchingConfigChanged,
    );
    bindSlider(
        "rng-rm-lipschitz-resolution",
        "val-rm-lipschitz-resolution",
        experimentState.rayMarchingConfig,
        "lipschitzGridResolution",
        true,
    );
    bindSlider(
        "rng-rm-lipschitz-samples",
        "val-rm-lipschitz-samples",
        experimentState.rayMarchingConfig,
        "lipschitzSamplesPerAxis",
        true,
    );
    bindSlider(
        "rng-rm-lipschitz-safety",
        "val-rm-lipschitz-safety",
        experimentState.rayMarchingConfig,
        "lipschitzSafetyFactor",
    );

    const btnApplyLipschitz = document.getElementById(
        "btn-apply-lipschitz",
    ) as HTMLButtonElement;
    if (btnApplyLipschitz) {
        btnApplyLipschitz.addEventListener("click", () => {
            renderer.rebuildLipschitzGrid();
        });
    }

    const btnCaptureMetrics = document.getElementById(
        "btn-capture-metrics",
    ) as HTMLButtonElement;
    if (btnCaptureMetrics) {
        btnCaptureMetrics.addEventListener("click", () => {
            renderer.captureMetrics();
        });
    }

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
