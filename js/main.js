let state = null;let L = 0;let K = 0;
let cubeRule = [[], [], [], [], [], [], [], [], []];
let faceRule = [[], [], [], [], [], [], []];
let edgeRule = [[], [], [], [], [], [], []];
let flipP = .9;
let symmetricAdd = 1.;
let caterrainsize = 0;
window.onload = async () => {
    const urlParams = new URLSearchParams(window.location.search).entries();
    const paramList  = {};
    for (pair of urlParams) {
        paramList[pair[0]] = pair[1];
    }

    const size = "size" in paramList ? paramList["size"] : 128;

    if ("seed" in paramList) {document.querySelector("#rule-seed").value = paramList["seed"];}

    //UI to add:
    //postprocessing - aces, reinhard, gamma, exposure
    const flags = {"changed": true, "lock-input": false};
    initUI(flags, {"setScene": setScene}, size);

    let vox = await voxels(document.querySelector("canvas"), size, 
    {profiling: false, "subvoxel-dimension": 3}
    );

    vox.setFOV(Math.PI / 4.);

    if (vox.adapterInfo) {
        const str = vox.adapterInfo["description"];
        let outstr = ""; const max_size = 21;
        if (str) {
            for (var i = 0; i < max_size - 3; i++) {
                if (str[i]) outstr += str[i];
            }
            if (str.length > max_size) {
                outstr += "...";
            } else {
                if (str[max_size - 2]) outstr += str[max_size - 2];
                if (str[max_size - 1]) outstr += str[max_size - 1];
                if (str[max_size - 0]) outstr += str[max_size - 0];
            }
        } else {
            outstr = "no info found";
        }
        document.querySelector("#adapter-info").innerHTML = outstr;
    }
    
    //set voxel materials
    const full = [
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    ];
    const menger = [
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ],
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
    ];
    const cross = [
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ];
    const tubeUp = [
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1],
        ],
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    ];
    
    const voxelMaterials = [full, menger, cross, tubeUp];

    

    //set caterrain parameters
    L = {"32": 5, "64": 6, "128": 7, "256": 8, "512": 9}["" + size]; caterrainsize = Math.pow(2, L) + 1; K = Math.floor(Math.pow(2, L)) + 1;

    document.querySelector("#rule-seed").oninput();
    document.querySelector("#regenerate-scene").onclick();

    document.querySelector("#save-image").onclick = () => {
        vox.downloadImage();
    };
    
    //called by UI functions when scene is done being built
    async function setScene() {
        for (var i = 0; i < size; i++) {
            for (var j = 0; j < size; j++) {
                for (var k = 0; k < size; k++) {
                    vox.setVoxel(i, j, k, state[i][j][k]);
                }
            }
        }
        await vox.uploadScene();
        vox.setReset();
    }

    //movement / interaction with scene
    let theta = Math.PI / 4.;
    let phi = -Math.PI / 8.;

    let position = [
        Math.cos(theta + Math.PI) * Math.cos(phi) * size * 2.5 + size / 2,
        Math.sin(theta + Math.PI) * Math.cos(phi) * size * 2.5 + size / 2,
        Math.sin(-phi) * size * 2.5 + size / 3
    ];
    let mousevelocity = {x: 0., y: 0.};
    let deltaX = 0.; let deltaY = 0.;
    document.querySelector("#render-target").addEventListener("mousedown", (e) => {
        const mouse = {x: e.clientX, y: e.clientY};
        if (flags["lock-input"]) return;
        mousevelocity = {x: 0., y: 0.};
        function mouseMove(e) {
            deltaX += e.clientX - mouse.x; deltaY += e.clientY - mouse.y;
            mouse.x = e.clientX; mouse.y = e.clientY;
        }
        document.querySelector("#render-target").addEventListener("mousemove", mouseMove);
        document.body.addEventListener("mouseup", () => {
            document.querySelector("#render-target").removeEventListener("mousemove", mouseMove);
        }, {once : true});
    });

    let positionvelocity = [0., 0., 0.];
    let keyinputs = [];
    document.addEventListener("keydown", (e) => {
        if (flags["lock-input"]) return;
        keyinputs.push(e.keyCode);
    });

    let then = Date.now() * .001;
    async function frame() {
        let now = Date.now() * .001;
        const delta = now - then;
        then = now;

        if (deltaX != 0 || deltaY != 0) {vox.setReset()};
        theta -= deltaX * .002;
        phi   -= deltaY * .002;
        phi = Math.min(Math.max(phi, .99 * -Math.PI / 2.), .99 * Math.PI / 2.);

        for (var x in keyinputs) {
            let speed = 1000.;
            switch (keyinputs[x]) {
                case 87:
                    positionvelocity[0] += Math.cos(theta) * Math.cos(phi) * delta * speed;
                    positionvelocity[1] += Math.sin(theta) * Math.cos(phi) * delta * speed;
                    positionvelocity[2] +=                   Math.sin(phi) * delta * speed;
                    break;
                case 83:
                    positionvelocity[0] -= Math.cos(theta) * Math.cos(phi) * delta * speed;
                    positionvelocity[1] -= Math.sin(theta) * Math.cos(phi) * delta * speed;
                    positionvelocity[2] -=                   Math.sin(phi) * delta * speed;
                    break;
                case 68:
                    positionvelocity[0] += Math.cos(theta - Math.PI / 2.) * delta * speed;
                    positionvelocity[1] += Math.sin(theta - Math.PI / 2.) * delta * speed;
                    positionvelocity[2] += 0.;
                    break;
                case 65:
                    positionvelocity[0] -= Math.cos(theta - Math.PI / 2.) * delta * speed;
                    positionvelocity[1] -= Math.sin(theta - Math.PI / 2.) * delta * speed;
                    positionvelocity[2] -= 0.;
                    break;
                case 32:
                    positionvelocity[2] += speed * delta;
                    break;
                case 16:
                    positionvelocity[2] -= speed * delta;
                    break;
            }
        }

        const speed = parseFloat(document.querySelector("#speed-slider").value);
        position[0] += positionvelocity[0] * delta * speed;
        position[1] += positionvelocity[1] * delta * speed;
        position[2] += positionvelocity[2] * delta * speed;

        if (
            Math.abs(positionvelocity[0]) > .2 ||
            Math.abs(positionvelocity[1]) > .2 ||
            Math.abs(positionvelocity[2]) > .2
            ) {vox.setReset();}
        else {
            mousevelocity.x = 0; mousevelocity.y = 0;
            positionvelocity = [0, 0, 0];
        }

        if (flags["changed"]) {
            vox.uploadRenderSettings(getUIValues());
        }
        flags["changed"] = false;
        if (flags["material"]) {
            vox.setMaterial(0, hexToRGB(document.querySelector("#color-0").value), [0., 0., 0., 0.], voxelMaterials[document.querySelector("#type-0").value]);
            vox.setMaterial(1, hexToRGB(document.querySelector("#color-1").value), [0., 0., 0., 0.], voxelMaterials[document.querySelector("#type-1").value]);
            vox.uploadMaterials();

            vox.setReset();
        }
        flags["material"] = false;

        positionvelocity[0] -= positionvelocity[0] * delta * 5.;
        positionvelocity[1] -= positionvelocity[1] * delta * 5.;
        positionvelocity[2] -= positionvelocity[2] * delta * 5.;
 
        keyinputs = [];
        deltaX = 0.; deltaY = 0.;

        vox.setPosition(position);
        vox.setLookAt([
            position[0] + Math.cos(theta) * Math.cos(phi),
            position[1] + Math.sin(theta) * Math.cos(phi),
            position[2] + Math.sin(phi)
        ]);

        let a = await vox.frame();

        if ("profiling" in a) {
            document.querySelector("#gbuffer").innerHTML = `----gbuffer (ms): ${a["profiling"]["gbuffer"]}`;
        } else {
            await a["done"];
        }
        document.querySelector("#samples-count").innerHTML = `${a["samples"]}`;
        window.requestAnimationFrame(frame);
    }
    frame();
};