async function voxels(canvas, size, options) {
    const adapter = await navigator.gpu?.requestAdapter();

    const reqs = [];
    if ("profiling" in options && options["profiling"] == true) {reqs.push("timestamp-query");}
    
    const device = await adapter?.requestDevice({requiredFeatures: reqs});
    if (!device) {console.error("browser does not support webGPU"); alert("browser does not support webGPU"); return null;}
    
    const context = canvas.getContext("webgpu");
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({device, format: presentationFormat});

    let width, height; let fReset = false;
    new ResizeObserver(onResize).observe(canvas);

    //chrome.exe --disable-dawn-features=disallow_unsafe_apis
    size = (size in {"32":null,"64":null,"128":null,"256":null,"512":null}) ? size : (() => {console.warning(`VOXELS ERROR: size ${size} is invalid. size must be power of two.`); return 64;})();
    const sceneWidth = size; const sceneWidthO2 = sceneWidth / 2;

    let subvoxels = {};
    if ("subvoxel-dimension" in options) {
        subvoxels["width"] = options["subvoxel-dimension"];
        if (subvoxels["width"] < 1) {
            console.error("< 1 value for subvoxel dimension");
            return {};
        }
        if (subvoxels["width"] > 12) {
            console.error("subvoxel dimension is too large");
            return {};
        }
    } else {
        subvoxels["width"] = 1;
    }
    {
        const numBits = subvoxels["width"] * subvoxels["width"] * subvoxels["width"];
        const numU32s = Math.ceil(numBits / 32);
        const numVec4 = Math.ceil(numU32s /  4);
        subvoxels["masklength"] = numVec4;
    }
    const subvoxelWidth = subvoxels["width"];
    const subvoxelMaskSize = subvoxels["masklength"];
    const bytesPerMaterial = 32 + 16 * subvoxelMaskSize;
    

    //---------------- Shader Code For all Pipelines ----------------//
    const structShaderCode = /* wgsl */`
    struct UBO {
        screenSize : vec2f,
        other : vec2f,
        invProj0 : vec4f,
        invProj1 : vec4f,
        invProj2 : vec4f,
        invProj3 : vec4f,
        invView0 : vec4f,
        invView1 : vec4f,
        invView2 : vec4f,
        invView3 : vec4f,
        lastMVM0 : vec4f,
        lastMVM1 : vec4f,
        lastMVM2 : vec4f,
        lastMVM3 : vec4f,
        lastPos  : vec4f,
        jitter   : vec2f
    };

    struct RenderSettings {
        focalDistance :   f32,
        aperture      :   f32,
        others0       : vec2f,
        sun           : vec4f,
        sky           : vec4f,
        azimuth       :   f32,
        zenith        :   f32,
        tonemapping   :   f32
    }

    struct VoxelMaterial {
        color_r : vec4f,
        others  : vec4f,
        mask : array<vec4u, ${subvoxelMaskSize}>
    };

    

    const Pi      = 3.14159265358979323846;
    const InvPi   = 0.31830988618379067154;
    const Inv2Pi  = 0.15915494309189533577;
    const Inv4Pi  = 0.07957747154594766788;
    const PiOver2 = 1.57079632679489661923;
    const PiOver4 = 0.78539816339744830961;
    const Sqrt2   = 1.41421356237309504880;

    //converts from a pixel coordinate (in [0, 1]) to a ray direction
    //based on the supplied inverse projection and inverse view matrices
    //based on: https://www.shadertoy.com/view/ltyXWh
    fn rayDir(coord : vec2f) -> vec3f {
        var nds : vec4f = vec4f(coord * vec2f(2., -2.) - vec2f(1., -1.), -1., 1.);
        var dirEye : vec4f = mat4x4<f32>(
            uniforms.invProj0, 
            uniforms.invProj1, 
            uniforms.invProj2, 
            uniforms.invProj3
            ) * nds;
        dirEye.w = 0.;
        var dirWorld : vec4f = mat4x4<f32>(
            uniforms.invView0,
            uniforms.invView1,
            uniforms.invView2,
            uniforms.invView3
        ) * dirEye;
        return normalize(dirWorld.xyz);
    }

    //fn rayDirHalf(coord : vec)

    //GPU hashes from: https://www.shadertoy.com/view/XlycWh
    var<private> bSeed : f32 = 0.f;
    fn baseHash(p : vec2u) -> u32 {
        var p2 : vec2u = 1103515245u*((p >> vec2u(1u))^(p.yx));
        var h32 : u32 = 1103515245u*((p2.x)^(p2.y>>3u));
        return h32^(h32 >> 16u);
    }
    fn initSeed(coord : vec2u) {
        bSeed = f32(baseHash(coord)) / f32(0xffffffffu) + uniforms.other.y * .008;// + f32(bitcast<u32>(uniforms.forward.a)) * .008;
    }
    fn rand2() -> vec2f {
        var n : u32 = baseHash(bitcast<vec2u>(vec2f(bSeed + 1., bSeed + 2.)));
        bSeed += 2.;
        var rz : vec2u = vec2u(n, n * 48271u);
        return vec2f(rz.xy & vec2u(0x7fffffffu))/f32(0x7fffffff);
    }

    //from: pbrt
    fn cosineSampleHemisphere() -> vec3f {
        var d : vec2f = uniformSampleDisk();
        var z : f32 = sqrt(max(0., 1. - d.x * d.x - d.y * d.y));
        return vec3f(d.xy, z);
    }
    fn uniformSampleDisk() -> vec2f {
        var r2 : vec2f = rand2();
        var r : f32 = sqrt(max(r2.x, 0.));
        var theta : f32 = 2. * Pi * r2.y;
        return vec2f(r * cos(theta), r * sin(theta));
    }`;

    //---------------- Shader Code For Ray Tracing ----------------//
    const traceFunctions = /* wgsl */ `
    struct RayHit {
        norm : vec3f,
        dist :   f32,
        bhit :  bool,
        umat :   u32
    };
    //projects a rays position within the voxel world bounds
    //o is the projected position, returns true if hits bounds
    //from: https://gist.github.com/DomNomNom/46bb1ce47f68d255fd5d
    fn projectPoint(o : ptr<function, vec3f>, d : vec3f) -> bool {
        var idir : vec3f = 1. / d;
        var tmin : vec3f = (-*o) * idir;
        var tmax : vec3f = (vec3f(${sceneWidth}.) - *o) * idir;
        var tone : vec3f = min(tmin, tmax);
        var ttwo : vec3f = max(tmin, tmax);
        var dist : f32   = max(max(tone.x, tone.y), tone.z);
        var dis2 : f32   = min(min(ttwo.x, ttwo.y), ttwo.z);
        var eps = .0005;
        (*o) = (*o) + (max(dist, 0.)) * d + sign(d) * eps * vec3f(vec3f(dist) == vec3f(dist));

        return dist < dis2 && dis2 > 0.;
    }

    fn trace(o : vec3f, d : vec3f, startlow : bool) -> RayHit {
        var returned : RayHit;

        ${
            (() => {
                const level = {"32": 5, "64": 6, "128": 7, "256": 8, "512": 9}["" + sceneWidth];
                return /*wgsl*/ `
                var level : u32 = ${level - 1}u;
                `;
            })()
        }

        var offset: i32 = ${sceneWidth - 2};

        if (startlow) {level = 0u; offset = 0;}

        var vsize : i32 = 1 << level;
        var vdim  : i32 = ${sceneWidth} >> level;

        var ipos : vec3i = vec3i(floor(o)) >> vec3u(level);

        var idir = abs(1. / d) * f32(vsize);
        var dist : vec3f = (sign(d) * (vec3f(ipos) - o * (1. / f32(vsize))) + sign(d) * .5 + .5) * idir;
        var step : vec3i = vec3i(sign(d));
        var descentmask : vec3i = vec3i(vec3f(d < vec3f(0.)));

        if (any(ipos < vec3i(0)) || any(ipos >= vec3i(vdim))) {return returned;}

        var eps = sign(d) * .0005;
        var bhit : bool = false;
        var mask : vec3<bool>;
        var i : i32 = 0;
        for (; i < 1000; i++) {
            var node : u32 = textureLoad(octree, (ipos >> vec3u(1u)) + vec3i(0, 0, offset), 0).x;
            
            var octant : vec3i = ipos & vec3i(1);
            var octmask : u32 = u32(dot(octant, vec3i(1, 2, 4)));

            if ((node & (1u << octmask)) != 0u) {
                var tmin : f32 = max(dot(dist - idir, vec3f(mask)), 0.);
                var wpos = o + d * tmin + eps * vec3f(vec3f(mask));
                ipos = vec3i(floor(wpos));
                if (level == 0u) {
                    var subvoxeldim : i32 = ${subvoxelWidth};

                    octant = ipos & vec3i(1);

                    var octidx : u32 = u32(dot(octant, vec3i(1, 2, 0)));
                    var material : u32 = textureLoad(scene, ipos >> vec3u(1u), 0)[octant.z] >> (octidx * 8u);
                    material &= 0xffu;

                    var subvoxelmask = materials[material - 1u].mask;

                    var subpos : vec3f = fract(wpos) * f32(subvoxeldim);
                    var subipos = vec3i(floor(subpos));

                    var subdist : vec3f = (sign(d) * (vec3f(subipos) - subpos) + sign(d) * .5 + .5) * idir;
                    var submask : vec3<bool> = mask;
            
                    for (var j = 0; j < 20; j++) {
                        if ((any(subipos < vec3i(0)) || any(subipos >= vec3i(subvoxeldim)))) {break;}
                        var maskidx : u32 = dot(vec3u(subipos), vec3u(1u, u32(subvoxeldim), u32(subvoxeldim * subvoxeldim)));
                        var subfilled : bool = 0u != ((1u << (maskidx % 32u)) & subvoxelmask[maskidx / 128u][(maskidx % 128u) / 32u]);
                        
                        if (subfilled) {
                            bhit = true;
                            break;
                        }
                        
                        submask = subdist <= min(subdist.zxy, subdist.yzx);
                        subipos += step * vec3i(vec3f(submask));
                        subdist += idir * vec3f(submask);
                    }
                    

                    if (bhit) {
                        var sdir : vec3f = sign(d) * idir;
                        var ts   : vec3f = (vec3f(ipos) + vec3f(subipos + descentmask) / f32(subvoxeldim) - (o)) * sdir;
                        var thit :   f32 = max(max(ts.x, 0.), max(ts.y, ts.z));
                        var norm : vec3f = vec3f(vec3f(thit) == vec3f(ts)) * -sign(d);

                        //hacky workaround for voxels on the edge of the scene
                        //would be better to just do an actual AABB or something
                        if (thit == 0.) {
                            var ts2 : vec3f = (vec3f(descentmask) * ${sceneWidth}. - o) * sdir;
                            norm = vec3f(vec3f(max(ts2.x, max(ts2.y, ts2.z))) == ts2) * -sign(d);
                        }

                        returned.norm = norm; 
                        returned.dist = thit; 
                        returned.bhit = true;
                        returned.umat = material;
                        break;
                    }
                } else {
                    level--;

                    offset -= vdim;
                    vdim  = vdim  << 1u;

                    ipos = ipos >> vec3u(level);
                    octant = ipos & vec3i(1);

                    var changemask = vec3f(octant == descentmask);

                    idir *= .5;

                    dist -= changemask * idir;

                    continue;
                }
            }

            mask = dist <= min(dist.zxy, dist.yzx);

            ipos += vec3i(vec3f(mask)) * step;

            var exited : vec3<bool> = ((vec3i(1) - descentmask) == octant);
            var ascend : bool = any(mask & exited);
            if (ascend || node == 0u) {
                dist += vec3f(!exited) * idir;
                idir *= 2.;
                level++;

                mask &= vec3<bool>(ascend);

                ipos = ipos >> vec3u(1u);
                vdim  = vdim  >> 1u;

                if (any(ipos < vec3i(0)) || any(ipos >= vec3i(vdim))) {break;}
                offset += vdim;
            }
            
            dist += vec3f(mask) * idir;
        }

        return returned;
    }`;

    //---------------- Vertex Shader for Pipelines ----------------//
    const vsShaderCode = /* wgsl */ `
    @vertex
    fn vs(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4f {
        switch(vertexIndex) {
            case 0u: {
                return vec4f(1., 1., 0., 1.);}
            case 1u: {
                return vec4f(-1., 1., 0., 1.);}
            case 2u: {
                return vec4f(-1., -1., 0., 1.);}
            case 3u: {
                return vec4f(1., -1., 0., 1.);}
            case 4u: {
                return vec4f(1., 1., 0., 1.);}
            case 5u: {
                return vec4f(-1., -1., 0., 1.);}
            default: {
                return vec4f(0., 0., 0., 0.);}
        }
    }`;

    //---------------- Copy Final Result to Canvas ----------------//
    const fsShaderCode = /* wgsl */`
    @group(0) @binding(0) var<uniform> uniforms : UBO;
    @group(0) @binding(1) var gbuffer : texture_2d<f32>;
    @group(0) @binding(2) var finalImage : texture_2d<f32>;
    @group(0) @binding(3) var<uniform> settings : RenderSettings;

    fn aces(x : vec3f) -> vec3f {
        return clamp(
            x * (2.51 * x + .03) / (x * (2.43 * x + .59) + .14), 
            vec3f(0.), 
            vec3f(1.)
        );
    }

    fn reinhard(z : vec3f) -> vec3f {
        return z / vec3f(1. + dot(z, vec3f(.2126, .7152, .0722)));
    }

    @fragment
    fn fs(@builtin(position) fragCoord : vec4f) -> @location(0) vec4f {
        var col : vec3f = textureLoad(finalImage, vec2i(fragCoord.xy), 0).xyz;
        if (settings.tonemapping == 0.) {
            col = aces(col);
        }
        if (settings.tonemapping == 1.) {
            col = reinhard(col);
        }
        col = pow(col, vec3f(1. / 2.2));
        return vec4f(col, 1.);
    }`;

    //---------------- Ray Trace the G Buffer ----------------//
    const gbShaderCode = /* wgsl */ `
    @group(0) @binding(0) var<uniform> uniforms : UBO;
    @group(0) @binding(1) var scene : texture_3d<u32>;
    @group(0) @binding(2) var octree: texture_3d<u32>;
    @group(0) @binding(3) var<uniform> materials : array<VoxelMaterial, 256>;
    @group(0) @binding(4) var accum : texture_2d<f32>;
    @group(0) @binding(5) var<uniform> settings : RenderSettings;

    struct RenderResult {
        @location(0) gbuffer : f32,
        @location(1) pathtrace : vec4f,
    }
    
    @fragment
    fn fs(@builtin(position) fragCoord : vec4f) -> RenderResult {
        initSeed(vec2u(fragCoord.xy));

        var aa = rand2(); aa = vec2f(cos(aa.x * 2. * Pi), sin(aa.x * 2. * Pi)) * aa.y * .5;

        var o : vec3f = uniforms.invView3.xyz;
        var d : vec3f = rayDir((fragCoord.xy + vec2f(.5, .5) + aa) / uniforms.screenSize);

        var forward: vec3f = normalize(uniforms.invView2.xyz * vec3f(1., 1., -1.));
        var ortho1 : vec3f;
        if (abs(forward.x) > abs(forward.y)) {
            ortho1 = vec3f(-forward.y, forward.x, 0.);
        } else {
            ortho1 = vec3f(0., -forward.z, forward.y);
        }
        ortho1 = normalize(ortho1);
        var ortho2 : vec3f = normalize(cross(forward, ortho1));

        var r2 = rand2(); r2 = vec2f(cos(r2.x * 2. * Pi), sin(r2.x * 2. * Pi)) * r2.y * settings.aperture;

        var focaltarget = o + d * settings.focalDistance;

        o += r2.x * ortho1 + r2.y * ortho2;
        d = normalize(focaltarget - o);
        d = select(vec3f(-1.), vec3f(1.), d > vec3f(0.)) * max(abs(d), vec3f(.0001));

        var returned : RenderResult;
        
        var col : vec3f = pt(o, d);

        var accumulated = (col + textureLoad(accum, vec2i(fragCoord.xy), 0).xyz * uniforms.other.y) / (uniforms.other.y + 1.);
        returned.pathtrace = vec4f(accumulated, 1.);

        return returned;
    }

    fn coneSample(d : vec3f, e : f32) -> vec3f {
        var o1 : vec3f;
        if (abs(d.x) > abs(d.y)) {
            o1 = vec3f(-d.y, d.x, 0.);
        } else {
            o1 = vec3f(0., -d.z, d.y);
        }
        o1 = normalize(o1); var o2 = normalize(cross(o1, d));

        var r : vec2f = rand2();
        var r2: vec2f = vec2f(r.x * 2. * Pi, 1. - r.y * e);
        var m1:   f32 = sqrt(max(0., 1. - r2.y * r2.y));
        return cos(r2.x) * m1 * o1 + sin(r2.y) * m1 * o2 + r2.y * d;
    }

    fn pt(oi : vec3f, di : vec3f) -> vec3f {
        var o : vec3f = oi;
        var d : vec3f = di;

        projectPoint(&o, d);

        var t : vec3f = vec3f(0.);
        var b : vec3f = vec3f(1.);

        var skycol : vec3f = pow(settings.sky.xyz, vec3f(2.2)) * settings.sky.a;//vec3f(0.02) * 8.;//vec3f(.5, .7, 1.);
        var suncol : vec3f = pow(settings.sun.xyz, vec3f(2.2)) * settings.sun.a;//vec3f(.1) * 8.;//vec3f(8.1, 6., 4.2) * .15;
        var sundir : vec3f = normalize(
            vec3f(
                cos(settings.azimuth) * cos(settings.zenith),
                sin(settings.azimuth) * cos(settings.zenith),
                sin(settings.zenith)
            ));

        for (var i : i32 = 0; i < 3; i++) {
            var result : RayHit = trace(o, d, i != 0);

            if (!result.bhit) {t += b * skycol; break;}

            var material : VoxelMaterial = materials[result.umat - 1u];

            if (material.others.x > 0.) {
                t += b * material.others.x * pow(material.color_r.xyz, vec3f(2.2));
                break;
            }

            var o1 : vec3f; var o2 : vec3f;
            if (abs(result.norm.x) == 1.) {
                o1 = vec3f(0., 1., 0.); o2 = vec3f(0., 0., 1.);
            }
            if (abs(result.norm.y) == 1.) {
                o1 = vec3f(1., 0., 0.); o2 = vec3f(0., 0., 1.);
            }
            if (abs(result.norm.z) == 1.) {
                o1 = vec3f(1., 0., 0.); o2 = vec3f(0., 1., 0.);
            }

            var wi : vec3f = cosineSampleHemisphere();

            var pdf :   f32 = max(wi.z, .001) * InvPi;
            var brdf: vec3f = vec3f(.4) * InvPi;

            b *= pow(material.color_r.xyz, vec3f(2.2));

            o = o + d * result.dist + result.norm * .001;
            d = wi.x * o1 + wi.y * o2 + wi.z * result.norm;

            {//sample direct light
                var dir : vec3f = coneSample(sundir, .0005);
                var dif :   f32 = dot(result.norm, dir);
                if (dif > 0.) {
                    var sunres : RayHit = trace(o, dir, true);
                    if (!sunres.bhit) {
                        t += b * suncol * dif;
                    }
                }
            }
        }

        return t;
    }`;

    //---------------- Editing & Rebuilding Acceleration Structure ----------------//
    //-------- Textures/Buffers that are used for Octree --------//
    //each voxel is a byte, so there can be 256 voxels in a palette
    const CPUSceneBuffer = new ArrayBuffer(sceneWidth * sceneWidth * sceneWidth);
    const CPUSceneArView = new Uint8Array(CPUSceneBuffer);

    //Textures that the rest of the pipeline will see
    const SCENE_TEXTURE = device.createTexture({
        size: [sceneWidthO2, sceneWidthO2, sceneWidthO2],
        format: "rg32uint",
        dimension: "3d",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });
    const OCTREE_TEXTURE = device.createTexture({
        size: [sceneWidthO2, sceneWidthO2, sceneWidth],
        format: "r8uint",
        dimension: "3d",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
    });

    //since each voxel has a byte to describe its material, we have 255 materials
    //each is stored as a uniform with a roughness, color, subvoxel mask, etc.
    const MaterialBuffer = device.createBuffer({
        size: bytesPerMaterial * 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const CPUMaterialBuffer = new ArrayBuffer(bytesPerMaterial * 256);

    //constants used to size and align buffers when copying to 3D textures
    const finalByteWidth = Math.ceil(sceneWidth / 256) * 256;

    //-------- Octree Generation Shader(s) Code --------//
    const ogShaderCode = /* wgsl */ `
    struct OPT {
        first   : u32,
        dim     : u32,
        offset  : u32,
        loffset : u32
    };
    @group(0) @binding(0) var<uniform> options : OPT;
    @group(0) @binding(1) var scene : texture_3d<u32>;
    @group(0) @binding(2) var<storage, read_write> intermediate : array<u32>;
    @group(0) @binding(3) var<storage, read_write> octree : array<u32>;

    //creates the next layer of the octree from the previous layer
    @compute @workgroup_size(4, 4, 4)
    fn main(@builtin(global_invocation_id) global_id : vec3u) {
        if (any(global_id >= vec3u(options.dim))) {return;}
        var outvalue : u32 = 0u;
        if (options.first != 0u) {
            var base : vec3u = global_id * 2u;
            for (var i : u32 = 0u; i < 2u; i++) {
                for (var j : u32 = 0u; j < 2u; j++) {
                    for (var k : u32 = 0u; k < 2u; k++) {
                        var offset : vec3u = vec3u(i, j, k);
                        var bindex :   u32 = dot(offset, vec3u(1u, 2u, 4u));
                        
                        var brick : vec2u = textureLoad(scene, (base + offset) >> vec3u(1u), 0).xy;
                        var bmask :   u32 = (0xffu) << ((bindex % 4u) * 8u);
                        var filled:  bool = (bmask & brick[bindex / 4u]) != 0u;
                        if (filled) {
                            outvalue |= 1u << bindex;
                        }
                    }
                }
            }
        } else {
            var base : vec3u = global_id * 2u;
            for (var i : u32 = 0u; i < 2u; i++) {
                for (var j : u32 = 0u; j < 2u; j++) {
                    for (var k : u32 = 0u; k < 2u; k++) {
                        var offset : vec3u = vec3u(i, j, k);
                        var bindex :   u32 = dot(offset, vec3u(1u, 2u, 4u));
            
                        var    pos : vec3u = base + offset;
                        var iindex :   u32 = dot(pos + vec3u(0u, 0u, options.loffset), vec3u(1u, ${sceneWidthO2}u, ${sceneWidthO2 * sceneWidthO2}u));
                        var lastval:   u32 = intermediate[iindex];
                        if (lastval != 0u) {
                            outvalue |= 1u << bindex;
                        }
                    }
                }
            }
        }

        var outindex : u32 = dot(global_id + vec3u(0u, 0u, options.offset), vec3u(1u, ${sceneWidthO2}u, ${sceneWidthO2 * sceneWidthO2}u));
        intermediate[outindex] = outvalue;
    }

    //takes the intermediate buffer and merges 4 bytes 
    //into a u32, with padding for buff->img conversion
    @compute @workgroup_size(1, 4, 4)
    fn fuse(@builtin(global_invocation_id) global_id : vec3u) {
        //global_id.y and global_id.z represent height & depth, while
        //global_id.x represents 4 bytes along a row (width)
        var pos : vec3u = global_id * vec3u(4u, 1u, 1u);
        if (any(pos > vec3u(${sceneWidthO2}u, ${sceneWidthO2}u, ${sceneWidth}u))) {return;}
        var idx : u32   = dot(pos, vec3u(1u, ${sceneWidthO2}u, ${sceneWidthO2 * sceneWidthO2}u));

        var outvalue : u32 = 
            (select(0u, intermediate[idx + 0u], (pos.x + 0u) < ${sceneWidthO2}u) <<  0u) |
            (select(0u, intermediate[idx + 1u], (pos.x + 1u) < ${sceneWidthO2}u) <<  8u) |
            (select(0u, intermediate[idx + 2u], (pos.x + 2u) < ${sceneWidthO2}u) << 16u) |
            (select(0u, intermediate[idx + 3u], (pos.x + 3u) < ${sceneWidthO2}u) << 24u) ;

        var outidx : u32 = dot(global_id, vec3u(1u, ${finalByteWidth / 4}u, ${(sceneWidthO2) * (finalByteWidth / 4)}u));
        octree[outidx]   = outvalue;
    }`;

    //-------- Create Octree Generation Pipelines/Layouts --------//
    const ogSM = device.createShaderModule({code: ogShaderCode});
    const ogBGLayout = device.createBindGroupLayout({
        label: "Octree Generation Bind Group Layout",
        entries: [
            {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.COMPUTE, texture: {sampleType: "uint", viewDimension: "3d", multisampled: false}},
            {binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
            {binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}}
        ]
    });
    const ogmPipeline = device.createComputePipeline({
        label: "Octree Generation Main Pipeline",
        layout: device.createPipelineLayout({bindGroupLayouts: [ogBGLayout]}),
        compute: {module: ogSM, entryPoint: "main"}
    });
    const ogfPipeline = device.createComputePipeline({
        label: "Octree Generation Fuse Pipeline",
        layout: device.createPipelineLayout({bindGroupLayouts: [ogBGLayout]}),
        compute: {module: ogSM, entryPoint: "fuse"}
    });

    //---------------- Scene/AS editing Exported Functions ----------------//
    //flush all of the changes from the CPU to the GPU
    async function uploadScene() {
        device.queue.writeTexture(
            {texture: SCENE_TEXTURE}, CPUSceneBuffer,
            {bytesPerRow: 8 * sceneWidthO2, rowsPerImage: sceneWidthO2},
            {width: sceneWidthO2, height: sceneWidthO2, depthOrArrayLayers: sceneWidthO2}
        );

        const intermediateBuffer = device.createBuffer({
            size: 4 * (sceneWidthO2 * sceneWidthO2 * sceneWidth),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const finalBuffer = device.createBuffer({
            size: finalByteWidth * sceneWidth * sceneWidthO2,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        const ogUniformBuffer = device.createBuffer({
            size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const ogBG = device.createBindGroup({
            label: "Octree Generation Bind Group",
            layout: ogBGLayout,
            entries: [
                {binding: 0, resource: {buffer: ogUniformBuffer}},
                {binding: 1, resource: SCENE_TEXTURE.createView()},
                {binding: 2, resource: {buffer: intermediateBuffer}},
                {binding: 3, resource: {buffer: finalBuffer}}
            ]
        });

        //actually build the octree
        let dimension = sceneWidthO2; let zOffset = 0; 
        let first = true; let lastZOffset = 0;
        let commandEncoder = device.createCommandEncoder();
        while (dimension >= 1) {
            device.queue.writeBuffer(ogUniformBuffer, 0, new Uint32Array(
                [first ? 1 : 0, dimension, zOffset, lastZOffset]
            ));
            //const commandEncoder = device.createCommandEncoder();
            const passEncoder    = commandEncoder.beginComputePass();
            
            passEncoder.setPipeline(ogmPipeline);
            passEncoder.setBindGroup(0, ogBG);
            const wgs = Math.max(dimension / 4, 1);
            passEncoder.dispatchWorkgroups(wgs, wgs, wgs);
            passEncoder.end();

            device.queue.submit([commandEncoder.finish()]);
            commandEncoder = device.createCommandEncoder();

            lastZOffset = zOffset; zOffset += dimension; dimension /= 2; first = false;
        }

        const passEncoder    = commandEncoder.beginComputePass();
        
        //pack the octree to be ready for a r8uint texture
        passEncoder.setPipeline(ogfPipeline);
        passEncoder.setBindGroup(0, ogBG);
        passEncoder.dispatchWorkgroups(
            Math.ceil(sceneWidth / 2),
            Math.ceil(sceneWidth / 8),
            Math.ceil(sceneWidth / 4)
        );
        passEncoder.end();
        
        //copy the octree to the target 3d texture
        commandEncoder.copyBufferToTexture({buffer: finalBuffer, bytesPerRow: finalByteWidth, rowsPerImage: sceneWidthO2}, {texture: OCTREE_TEXTURE}, {width: sceneWidthO2, height: sceneWidthO2, depthOrArrayLayers: sceneWidth});
        
        //submit all work to GPU
        device.queue.submit([commandEncoder.finish()]);
        if (device.queue.onSubmittedWorkDone) {
            await device.queue.onSubmittedWorkDone();
        }
        
        intermediateBuffer.destroy();
        finalBuffer.destroy();        
    }

    const sceneWidthO22 = sceneWidthO2 * sceneWidthO2;
    //Sets a voxel to be filled in the CPU buffer, does not
    //actually make any update on the GPU
    function setVoxel(x, y, z, material) {
        let idx = Math.floor(x * .5) + Math.floor(y * .5) * sceneWidthO2 + Math.floor(z * .5) * sceneWidthO22;
        idx = idx * 2;
        if (z % 2 == 1) {idx += 1;}
        CPUSceneArView[idx * 4 + (x % 2) + (y % 2) * 2] = material & 255;   
    }

    //Sets material in CPU Buffer, does not actually send
    //updates to the GPU, also applies gamma correction
    function setMaterial(index, color, other, subvoxel) {
        const dv = new DataView(CPUMaterialBuffer);

        dv.setFloat32(index * bytesPerMaterial + 0, color[0], true);
        dv.setFloat32(index * bytesPerMaterial + 4, color[1], true);
        dv.setFloat32(index * bytesPerMaterial + 8, color[2], true);
        dv.setFloat32(index * bytesPerMaterial + 12, 1., true);
        
        dv.setFloat32(index * bytesPerMaterial + 16, other[0], true);
        dv.setFloat32(index * bytesPerMaterial + 20, other[1], true);
        dv.setFloat32(index * bytesPerMaterial + 24, other[2], true);
        dv.setFloat32(index * bytesPerMaterial + 28, other[3], true);

        const u8view = new Uint8Array(CPUMaterialBuffer);
        for (var i = 0; i < subvoxelWidth; i++) {
            for (var j = 0; j < subvoxelWidth; j++) {
                for (var k = 0; k < subvoxelWidth; k++) {
                    const idx = i + j * subvoxelWidth + k * subvoxelWidth * subvoxelWidth;
                    u8view[index * bytesPerMaterial + 32 + Math.floor(idx / 8)] &= ~(1 << (idx % 8));
                    u8view[index * bytesPerMaterial + 32 + Math.floor(idx / 8)] |= (subvoxel[i][j][k] > 0 ? 1 << (idx % 8) : 0);
                }
            }
        }
    }

    //Actually sends the material values to the GPU to update them
    function uploadMaterials() {device.queue.writeBuffer(MaterialBuffer, 0, CPUMaterialBuffer, 0, 256 * bytesPerMaterial);}

    //---------------- Create Rendering Layouts/Pipelines ----------------//
    //-------- Create Rendering Buffers/Textures  --------//
    const uniformBufferSize = (16) * 15;
    const uniformBuffer = device.createBuffer({
        size: uniformBufferSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const renderSettingsBuffer = device.createBuffer({
        size: 4 * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    let gBufferResultTexture = null;
    let pathtraceResultTextures = [null, null];

    //-------- Final Fullscreen Copy --------//
    const fsBGLayout = device.createBindGroupLayout({
        label: "Full Screen Bind Group Layout",
        entries: [
            {binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {sampleType: "unfilterable-float", viewDimension: "2d", multisampled: false}},
            {binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {sampleType: "float", viewDimension: "2d", multisampled: false}},
            {binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}}
        ]
    });
    let fsBGs = [null, null];
    const fsShaderModule = device.createShaderModule({
        label: "Full Screen Shader Module", code: structShaderCode + vsShaderCode + fsShaderCode
    });
    const fsPipeline = device.createRenderPipeline({
        label: "Full Screen Render Pipeline",
        layout: device.createPipelineLayout({bindGroupLayouts: [fsBGLayout]}),
        vertex: {module: fsShaderModule, entryPoint: "vs"},
        fragment: {
            module: fsShaderModule, entryPoint: "fs",
            targets: [{format: presentationFormat}]
        }
    });
    const dlPipeline = device.createRenderPipeline({
        label: "Download Image Render Pipeline",
        layout: device.createPipelineLayout({bindGroupLayouts: [fsBGLayout]}),
        vertex: {module: fsShaderModule, entryPoint: "vs"},
        fragment: {
            module: fsShaderModule, entryPoint: "fs",
            targets: [{format: "rgba8unorm"}]
        }
    });

    //-------- Generate G Buffer @ Full Resolution --------//
    const gbBGLayout = device.createBindGroupLayout({
        label: "G Buffer Bind Group Layout",
        entries: [
            {binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
            {binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {sampleType: "uint", viewDimension: "3d", multisampled: false}},
            {binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {sampleType: "uint", viewDimension: "3d", multisampled: false}},
            {binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}},
            {binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {sampleType: "unfilterable-float", viewDimension: "2d", multisampled: false}},
            {binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: {type: "uniform"}}
        ]
    }); 
    let gbBGs = [null, null];
    const gbShaderModule = device.createShaderModule({
        label: "G Buffer Shader Module", code: structShaderCode + vsShaderCode + gbShaderCode + traceFunctions
    });
    const gbPipeline = device.createRenderPipeline({
        label: "G Buffer Render Pipeline",
        layout: device.createPipelineLayout({bindGroupLayouts: [gbBGLayout]}),
        vertex: {module: gbShaderModule, entryPoint: "vs"},
        fragment: {
            module: gbShaderModule, entryPoint: "fs",
            targets: [{format: "r32float"}, {format: "rgba16float"}]
        }
    });

    //---------------- Camera and Render States ----------------//
    const position = [0, 0, 0]; //the position of the camera
    const lookAt = [1, 0, 0];   //the point the camera looks at
    const camera = {
        "fov": Math.PI / 2.,    //camera FOV
        "scale": 1,             //pixels to render per screen pixel
    };

    //---------------- Utility Functions ----------------//
    //called when the target canvas is resized and on start
    //note: currently not exported, only for internal use
    function onResize() {
        fReset = true;
        //-------- Set Canvas Render Size --------//
        width = canvas.offsetWidth * camera["scale"];  
        height = canvas.offsetHeight * camera["scale"];
        canvas.width = width; canvas.height = height;
        if (width % 2 == 1) {width++;}
        if (height% 2 == 1) {height++;}

        //-------- Recreate Textures --------//
        if (gBufferResultTexture) gBufferResultTexture.destroy();
        gBufferResultTexture = device.createTexture({
            size: [width, height],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
        });
        if (pathtraceResultTextures[0]) pathtraceResultTextures[0].destroy();
        if (pathtraceResultTextures[1]) pathtraceResultTextures[1].destroy();
        pathtraceResultTextures = [
            device.createTexture({
                size: [width, height],
                format: "rgba16float",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
            }),
            device.createTexture({
                size: [width, height],
                format: "rgba16float",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
            })
        ];

        //-------- Recreate Bind Groups --------//
        gbBGs[0] = device.createBindGroup({
            label: "G Buffer Bind Group",
            layout: gbBGLayout,
            entries: [
                {binding: 0, resource: {buffer: uniformBuffer}},
                {binding: 1, resource: SCENE_TEXTURE.createView()},
                {binding: 2, resource:OCTREE_TEXTURE.createView()},
                {binding: 3, resource: {buffer: MaterialBuffer}},
                {binding: 4, resource: pathtraceResultTextures[1].createView()},
                {binding: 5, resource: {buffer: renderSettingsBuffer}}
            ]
        });
        gbBGs[1] = device.createBindGroup({
            label: "G Buffer Bind Group",
            layout: gbBGLayout,
            entries: [
                {binding: 0, resource: {buffer: uniformBuffer}},
                {binding: 1, resource: SCENE_TEXTURE.createView()},
                {binding: 2, resource:OCTREE_TEXTURE.createView()},
                {binding: 3, resource: {buffer: MaterialBuffer}},
                {binding: 4, resource: pathtraceResultTextures[0].createView()},
                {binding: 5, resource: {buffer: renderSettingsBuffer}}
            ]
        });

        fsBGs[0] = device.createBindGroup({
            label: "Full Screen Bind Group",
            layout: fsBGLayout,
            entries: [
                {binding: 0, resource: {buffer: uniformBuffer}},
                {binding: 1, resource: gBufferResultTexture.createView()},
                {binding: 2, resource: pathtraceResultTextures[0].createView()},
                {binding: 3, resource: {buffer: renderSettingsBuffer}}
            ]
        });
        fsBGs[1] = device.createBindGroup({
            label: "Full Screen Bind Group",
            layout: fsBGLayout,
            entries: [
                {binding: 0, resource: {buffer: uniformBuffer}},
                {binding: 1, resource: gBufferResultTexture.createView()},
                {binding: 2, resource: pathtraceResultTextures[1].createView()},
                {binding: 3, resource: {buffer: renderSettingsBuffer}}
            ]
        });
    }
    onResize();

    //if enabled, and query sets are enabled in browser, can
    //return performance information by stage in each frame
    const PROFILING_MODULE = {};
    if (options["profiling"]) {
        PROFILING_MODULE["capacity"] = 3;
        PROFILING_MODULE["querySet"] = device.createQuerySet({
            type: "timestamp", count: PROFILING_MODULE["capacity"],
        });
        PROFILING_MODULE["queryBuffer"] = device.createBuffer({
            size: 8 * PROFILING_MODULE["capacity"],
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        PROFILING_MODULE["readBuffer"] = device.createBuffer({
            size: 8 * PROFILING_MODULE["capacity"],
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
    }
    
    //---------------- Exported Functions ----------------//
    //called each frame, calls all every stage in the rendering pipeline
    let l = glMatrix.mat4.create();
    let o = [position[0], position[1], position[2]];
    let frames = 0;
    async function frame() {
        if (fReset) {frames = 0;}
        if (frames > 511) {return {done: true, samples: frames};}
        const pingpong     = frames % 2;
        //-------- Create and Write Inverse View and Projection Matrices --------//
        let t = glMatrix.mat4.create(); let v = glMatrix.mat4.create();
        glMatrix.mat4.lookAt(t, position, lookAt, [0, 0, 1]);
        let i = glMatrix.mat4.create(); let p = glMatrix.mat4.create();
        glMatrix.mat4.perspective(i, camera["fov"], width / height, .1, 100.);
        glMatrix.mat4.invert(v, t); glMatrix.mat4.invert(p, i);

        //-------- Create and Send GPU commands --------//
        const commandEncoder = device.createCommandEncoder();

        device.queue.writeBuffer(uniformBuffer, 0, new Float32Array(
            [
                width, height,
                0,//parseFloat(document.querySelector("#param-0").dataset.val),
                frames,
                p[0 ], p[1 ], p[2 ], p[3 ],
                p[4 ], p[5 ], p[6 ], p[7 ],
                p[8 ], p[9 ], p[10], p[11],
                p[12], p[13], p[14], p[15],
                v[0 ], v[1 ], v[2 ], v[3 ],
                v[4 ], v[5 ], v[6 ], v[7 ],
                v[8 ], v[9 ], v[10], v[11],
                v[12], v[13], v[14], v[15],
                l[0 ], l[1 ], l[2 ], l[3 ],
                l[4 ], l[5 ], l[6 ], l[7 ],
                l[8 ], l[9 ], l[10], l[11],
                l[12], l[13], l[14], l[15],
                o[ 0], o[ 1], o[ 2], 3.141,
            ]
        ));
        glMatrix.mat4.multiply(l, i, t);
        o = [position[0], position[1], position[2]];

        if (options["profiling"]) commandEncoder.writeTimestamp(PROFILING_MODULE["querySet"], 0);
        
        const gbPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: gBufferResultTexture.createView(),
                    clearValue: {r: 0., g: 0., b: 0., a: 0.},
                    loadOp: "clear", storeOp: "store"
                },
                {
                    view: pathtraceResultTextures[pingpong].createView(),
                    clearValue: {r: 0., g: 0., b: 0., a: 0.},
                    loadOp: "clear", storeOp: "store"
                }
            ]
        });
        gbPass.setPipeline(gbPipeline);
        gbPass.setBindGroup(0, gbBGs[pingpong]);
        gbPass.draw(6);
        gbPass.end();

        if (options["profiling"]) commandEncoder.writeTimestamp(PROFILING_MODULE["querySet"], 1);

        const fsPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context.getCurrentTexture().createView(),
                    clearValue: {r: 1., g: 0., b: 0., a: 1.},
                    loadOp: "clear", storeOp: "store"
                }
            ]
        });
        fsPass.setPipeline(fsPipeline);
        fsPass.setBindGroup(0, fsBGs[pingpong]);
        fsPass.draw(6);
        fsPass.end();

        if (options["profiling"]) commandEncoder.writeTimestamp(PROFILING_MODULE["querySet"], 2);
        if (options["profiling"]) commandEncoder.resolveQuerySet(PROFILING_MODULE["querySet"], 0, PROFILING_MODULE["capacity"], PROFILING_MODULE["queryBuffer"], 0);

        device.queue.submit([commandEncoder.finish()]);

        const returned = {};

        if (options["profiling"]) {
            const copyEncoder = device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(PROFILING_MODULE["queryBuffer"], 0, PROFILING_MODULE["readBuffer"], 0, 8 * PROFILING_MODULE["capacity"]);
            device.queue.submit([copyEncoder.finish()]);

            await PROFILING_MODULE["readBuffer"].mapAsync(GPUMapMode.READ);
            const buf = await PROFILING_MODULE["readBuffer"].getMappedRange();
            const vals = new BigInt64Array(buf);

            returned["profiling"] = {
                "gbuffer" : parseFloat(vals[1] - vals[0]) / 1000000.,
                "fragment": parseFloat(vals[2] - vals[1]) / 1000000.
            };
            PROFILING_MODULE["readBuffer"].unmap();
        }
        frames++;
        fReset = false;
        
        returned["done"] = device.queue.onSubmittedWorkDone;
        returned["samples"] = frames;

        return returned;
    }

    //downloads the image of the current frame
    async function downloadImage() {
        const resultImage = device.createTexture({
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
        });

        const byteWidth = Math.ceil((width * 4) / 256) * 256;

        const commandEncoder = device.createCommandEncoder();
        
        const drawPass = commandEncoder.beginRenderPass({
            colorAttachments: [
                {
                    view: resultImage.createView(),
                    clearValue: {r: 1, g: 0, b: 0, a: 1},
                    loadOp: "clear", storeOp: "store"
                }
            ]
        });

        drawPass.setPipeline(dlPipeline);
        drawPass.setBindGroup(0, fsBGs[0]);
        drawPass.draw(6);
        drawPass.end();

        device.queue.submit([commandEncoder.finish()]);

        const copyEncoder = device.createCommandEncoder();

        const dstBuffer = device.createBuffer({
            size: byteWidth * height,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        copyEncoder.copyTextureToBuffer({texture: resultImage}, {buffer: dstBuffer, bytesPerRow: byteWidth}, {width: width, height: height});

        device.queue.submit([copyEncoder.finish()]);

        await dstBuffer.mapAsync(GPUMapMode.READ);
        const buf = await dstBuffer.getMappedRange();
        const u8  = new Uint8Array(buf);

        const imageBuffer = new Uint8ClampedArray(width * height * 4);
        const w4 = width * 4;
        for (var j = 0; j < height; j++) {
            for (var i = 0; i < byteWidth; i++) {
                imageBuffer[j * w4 + i] = u8[j * byteWidth + i];
            }
        }

        dstBuffer.unmap();
        dstBuffer.destroy();
        resultImage.destroy();

        const c = document.createElement("canvas");
        const ctx = c.getContext("2d");
        c.width = width; c.height = height;
        const idata = ctx.createImageData(width, height);
        idata.data.set(imageBuffer);
        ctx.putImageData(idata, 0, 0);

        var link = document.createElement("a");
        link.download = "save.png";
        link.href = c.toDataURL();
        link.click();
    }

    //sets the position of the camera from an array of values
    function setPosition(arr) {
        position[0] = (!("0" in arr) || arr[0] == null) ? position[0] : arr[0];
        position[1] = (!("1" in arr) || arr[1] == null) ? position[1] : arr[1];
        position[2] = (!("2" in arr) || arr[2] == null) ? position[2] : arr[2];
    }

    //sets where the center of the camera should look at
    function setLookAt(arr) {
        lookAt[0] = (!("0" in arr) || arr[0] == null) ? lookAt[0] : arr[0];
        lookAt[1] = (!("1" in arr) || arr[1] == null) ? lookAt[1] : arr[1];
        lookAt[2] = (!("2" in arr) || arr[2] == null) ? lookAt[2] : arr[2];
    }

    //sets the camera projection matrix's field of view
    function setFOV(value) {
        camera["fov"] = Math.min(Math.max(.01, value), Math.PI - .01);
    }
    
    //sets the progressive renderer to reset accumulation
    function setReset() {fReset = true;}

    //sets render settings and flags a reset of accumulation
    function uploadRenderSettings(arr) {
        device.queue.writeBuffer(renderSettingsBuffer, 0, arr.buffer, 0, arr.byteLength);
        fReset = true;
    }

    return {frame, setPosition, setLookAt, setFOV, uploadScene, setVoxel, setMaterial, uploadMaterials, setReset, uploadRenderSettings, downloadImage, adapterInfo: await adapter.requestAdapterInfo()};
}