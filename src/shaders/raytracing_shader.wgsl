struct Ray {
    direction: vec3f,
    origin: vec3f,
}

struct Sphere {
    center: vec3f,
    color: vec3f,
    radius: f32,
}

struct ObjectData {
    spheres: array<Sphere>,
}

struct SceneData {
    cameraPos: vec3f,
    cameraForward: vec3f,
    cameraRight: vec3f,
    cameraUp: vec3f,
    sphereCount: f32,
}

struct RenderState {
    t: f32,
    color: vec3f,
    hit: bool,
}

@group(0) @binding(0) var colorBuffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> scene: SceneData;
@group(0) @binding(2) var<storage, read> objects: ObjectData;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) GlobalInvocationId: vec3u) {

    let screenSize: vec2i = vec2i(textureDimensions(colorBuffer));
    let screenPos: vec2i = vec2i(i32(GlobalInvocationId.x), i32(GlobalInvocationId.y));
    if (screenPos.x >= screenSize.x || screenPos.y >= screenSize.y) {
        return;
    }

    let horizontalCoeff: f32 = (f32(screenPos.x) - f32(screenSize.x) / 2.0) / f32(screenSize.x);
    let verticalCoeff: f32 = (f32(screenPos.y) - f32(screenSize.y) / 2.0) / f32(screenSize.x);

    var r: Ray;
    r.direction = normalize(scene.cameraForward + horizontalCoeff * scene.cameraRight + verticalCoeff * scene.cameraUp);
    r.origin = scene.cameraPos;

    var pixelColor: vec3f = rayColor(r);

    textureStore(colorBuffer, screenPos, vec4f(pixelColor, 1.0));
}

fn rayColor(ray: Ray) -> vec3f {
    // sky gradient -> blendedValue=(1−a) * startValue + a * endValue,
    let a: f32 = 0.5 * (-normalize(ray.direction).z + 1.0);
    var color: vec3f = (1.0 - a) * vec3f(1.0, 1.0, 1.0) + a * vec3f(0.5, 0.7, 1.0);

    var nearestHit: f32 = 9999;
    var hitSomething: bool = false;

    var renderState: RenderState;

    for (var i: u32 = 0; i < u32(scene.sphereCount); i++) {
        var newRenderState: RenderState = hitSphere(ray, objects.spheres[i], 0.001, nearestHit, renderState);

        if (newRenderState.hit) {
            nearestHit = newRenderState.t;
            renderState = newRenderState;
            hitSomething = true;
        }
    }

    if (hitSomething) {
        color = renderState.color;
    }
    return color;
}

fn hitSphere(ray: Ray, sphere: Sphere, tMin: f32, tMax: f32, oldRenderState: RenderState) -> RenderState {

    // c means the Center of sphere and o the ray Origin
    let co: vec3f = ray.origin - sphere.center;
    let a: f32 = dot(ray.direction, ray.direction);
    let b: f32 = 2.0 * dot(ray.direction, co);
    let c: f32 = dot(co, co) - sphere.radius * sphere.radius;
    let discriminant: f32 = b * b - 4.0 * a * c;

    var renderState: RenderState;

    if (discriminant > 0.0) {
        let t: f32 = (-b - sqrt(discriminant)) / (2.0 * a);

        if (t > tMin && t < tMax) {
            renderState.t = t;
            renderState.color = sphere.color;
            renderState.hit = true;
            return renderState;
        }
    }

    renderState.hit = false;
    renderState.color = oldRenderState.color;
    return renderState;
}