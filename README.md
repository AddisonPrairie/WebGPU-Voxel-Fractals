# WebGPU-Voxel-Fractals

This is an interactive voxel renderer written in WebGPU. It uses a JavaScript port of [Voxel Automata Terrain](https://bitbucket.org/BWerness/voxel-automata-terrain/src/master/) to generate 3D voxel fractals.
 The demo page is [here](https://addisonprairie.github.io/WebGPU-Voxel-Fractals/?size=128). For information on the controls, click the "help" button. NOTE: this demo requires WebGPU; check [here](https://caniuse.com/webgpu) whether your browser implements the WebGPU API. 

![](https://live.staticflickr.com/65535/53117430361_9ee921e373_z.jpg)<br>
(512^3 voxel fractal generated in JavaScript and rendered on the GPU using WebGPU)

While this demo is capable of running on integrated graphics, the performance is generally very slow. I would recommend running this on a browser set to use your discrete GPU. I developed and tested it on my laptop (iGPU: Ryzen 9 4900HS w/ Radeon Graphics, GPU: NVidia GeForce RTX 2060 with Max-Q design) and got reasonable results. All of the images on this page were rendered on my discrete GPU in about 10s @ 1024 samples / pixel.

![](https://live.staticflickr.com/65535/53117853205_ddae4961de_c.jpg)<br><br>
![](https://live.staticflickr.com/65535/53117447736_b4a2a23fe7_c.jpg)<br><br>
![](https://live.staticflickr.com/65535/53117446946_827ffd6cb8_c.jpg)<br><br>

The rendering code was written by me. It consists of a few fragment shaders for raytracing, sample accumulation, and post processing. It uses [glMatrix](https://github.com/toji/gl-matrix) for JS linear algebra operations. Other attributions for specific functions can be found in the source code.
