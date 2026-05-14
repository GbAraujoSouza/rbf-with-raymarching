export {};

const canvas = document.querySelector<HTMLCanvasElement>("#gpu-canvas");
const statusElement = document.querySelector<HTMLParagraphElement>("#status");

if (!canvas || !statusElement) {
  throw new Error("Missing required DOM nodes.");
}

const gpuCanvas = canvas;
const liveStatus = statusElement;

const setStatus = (message: string) => {
  liveStatus.textContent = message;
};

const resizeCanvas = () => {
  const devicePixelRatio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(gpuCanvas.clientWidth * devicePixelRatio));
  const height = Math.max(1, Math.floor(gpuCanvas.clientHeight * devicePixelRatio));

  if (gpuCanvas.width !== width || gpuCanvas.height !== height) {
    gpuCanvas.width = width;
    gpuCanvas.height = height;
  }
};

async function initWebGpu() {
  if (!("gpu" in navigator)) {
    setStatus("WebGPU is not available in this browser. Try a recent Chrome, Edge, or Safari Technology Preview.");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();

  if (!adapter) {
    setStatus("The browser supports WebGPU, but no GPU adapter was returned.");
    return;
  }

  const device = await adapter.requestDevice();
  const context = gpuCanvas.getContext("webgpu") as GPUCanvasContext | null;

  if (!context) {
    setStatus("Could not create a WebGPU canvas context.");
    return;
  }

  const format = navigator.gpu.getPreferredCanvasFormat();

  const configure = () => {
    resizeCanvas();
    context.configure({
      alphaMode: "opaque",
      device,
      format
    });
  };

  configure();
  window.addEventListener("resize", configure);

  const render = () => {
    const encoder = device.createCommandEncoder();
    const view = context.getCurrentTexture().createView();

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          clearValue: { r: 0.07, g: 0.36, b: 0.64, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
          view
        }
      ]
    });

    pass.end();
    device.queue.submit([encoder.finish()]);
  };

  render();
  setStatus("WebGPU initialized successfully.");
}

initWebGpu().catch((error: unknown) => {
  console.error(error);
  const message = error instanceof Error ? error.message : "Unknown error";
  setStatus(`Failed to initialize WebGPU: ${message}`);
});
