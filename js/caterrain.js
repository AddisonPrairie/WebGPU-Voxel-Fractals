function initArr() {
    state = [];

    for (var x = 0; x < caterrainsize; x++) {
        state[x] = [];
        for (var y = 0; y < caterrainsize; y++) {
            state[x][y] = [];
            for (var z = 0; z < caterrainsize; z++) {
                state[x][y][z] = 0.;
                
            }
        }
    }
}

function newRule(rng) {
    initArr();
    randomRule(Math.max(Math.min(rng(), .7), .3), rng);
}

async function newStateRandom() {
    initState();
    await evalState();
}

//port of:
//https://bitbucket.org/BWerness/voxel-automata-terrain/src/master/ThreeState3dBitbucket.pde
function evalCube(i, j, k, w) {
    if (i < 0 || j < 0 || k < 0 || i + w >= K || j + w >= K || k + w >= K) return;

    let idx1 =  (state[i][j][k] == 1 ? 1 : 0) + (state[i + w][j][k] == 1 ? 1 : 0) + (state[i + w][j + w][k] == 1 ? 1 : 0) +
                (state[i][j][k + w] == 1 ? 1 : 0) + (state[i + w][j][k + w] == 1 ? 1 : 0) + (state[i + w][j + w][k + w] == 1 ? 1 : 0) + 
                (state[i][j + w][k + w] == 1 ? 1 : 0) + (state[i][j][k + w] == 1 ? 1 : 0) + (state[i][j + w][k] == 1 ? 1 : 0);
    let idx2 =  (state[i][j][k] == 2 ? 1 : 0) + (state[i + w][j][k] == 2 ? 1 : 0) + (state[i + w][j + w][k] == 2 ? 1 : 0) +
                (state[i][j][k + w] == 2 ? 1 : 0) + (state[i + w][j][k + w] == 2 ? 1 : 0) + (state[i + w][j + w][k + w] == 2 ? 1 : 0) + 
                (state[i][j + w][k + w] == 2 ? 1 : 0) + (state[i][j][k + w] == 2 ? 1 : 0) + (state[i][j + w][k] == 2 ? 1 : 0);

    let w2 = Math.floor(w/2);
    state[i + w2][j + w2][k + w2] = cubeRule[idx1][idx2];

    if (state[i + w2][j + w2][k + w2] > 0 && Math.random() > flipP) {
        state[i + w2][j + w2][k + w2] = 3 - state[i + w2][j + w2][k + w2];
    }
}

function f1(i, j, k , w) {
    let w2 = Math.floor(w/2);
    if (i < 0 || j < 0 || k - w2 < 0 || i + w >= K || j + w >= K || k + w2 >= K) return;

    let idx1 =  (state[i][j][k] == 1 ? 1 : 0) + (state[i + w][j][k] == 1 ? 1 : 0) + (state[i][j + w][k] == 1 ? 1 : 0)  +
                (state[i + w][j + w][k] == 1 ? 1 : 0) + (state[i + w2][j + w2][k - w2] == 1 ? 1 : 0) + (state[i + w2][j + w2][k + w2] == 1 ? 1 : 0); 
    let idx2 =  (state[i][j][k] == 2 ? 1 : 0) + (state[i + w][j][k] == 2 ? 1 : 0) + (state[i][j + w][k] == 2 ? 1 : 0)  +
                (state[i + w][j + w][k] == 2 ? 1 : 0) + (state[i + w2][j + w2][k - w2] == 2 ? 1 : 0) + (state[i + w2][j + w2][k + w2] == 2 ? 1 : 0);

    state[i + w2][j + w2][k] = faceRule[idx1][idx2];

    if (state[i + w2][j + w2][k] > 0 && Math.random() > flipP) {
        state[i + w2][j + w2][k] = 3 - state[i + w2][j + w2][k];
    }
}

function f2(i, j, k , w) {
    let w2 = Math.floor(w/2);
    if (i < 0 || j - w2 < 0 || k < 0 || i + w >= K || j + w2 >= K || k + w >= K) return;

    let idx1 =  (state[i][j][k] == 1 ? 1 : 0) + (state[i + w][j][k] == 1 ? 1 : 0) + (state[i][j][k + w] == 1 ? 1 : 0)  +
                (state[i + w][j][k + w] == 1 ? 1 : 0) + (state[i + w2][j - w2][k + w2] == 1 ? 1 : 0) + (state[i + w2][j + w2][k + w2] == 1 ? 1 : 0); 
    let idx2 =  (state[i][j][k] == 2 ? 1 : 0) + (state[i + w][j][k] == 2 ? 1 : 0) + (state[i][j][k + w] == 2 ? 1 : 0)  +
                (state[i + w][j][k + w] == 2 ? 1 : 0) + (state[i + w2][j - w2][k + w2] == 2 ? 1 : 0) + (state[i + w2][j + w2][k + w2] == 2 ? 1 : 0); 

    state[i + w2][j][k + w2] = faceRule[idx1][idx2];

    if (state[i + w2][j][k + w2] > 0 && Math.random() > flipP) {
        state[i + w2][j][k + w2] = 3 - state[i + w2][j][k + w2];
    }
}

function f3(i, j, k , w) {
    let w2 = Math.floor(w/2);
    if (i - w/2 < 0 || j < 0 || k < 0 || i + w2 >= K || j + w >= K || k + w >= K) return;

    let idx1 =  (state[i][j][k] == 1 ? 1 : 0) + (state[i][j][k + w] == 1 ? 1 : 0) + (state[i][j + w][k] == 1 ? 1 : 0)  +
                (state[i][j + w][k + w] == 1 ? 1 : 0) + (state[i - w2][j + w2][k + w2] == 1 ? 1 : 0) + (state[i + w2][j + w2][k + w2] == 1 ? 1 : 0); 
    let idx2 =  (state[i][j][k] == 2 ? 1 : 0) + (state[i][j][k + w] == 2 ? 1 : 0) + (state[i][j + w][k] == 2 ? 1 : 0)  +
                (state[i][j + w][k + w] == 2 ? 1 : 0) + (state[i - w2][j + w2][k + w2] == 2 ? 1 : 0) + (state[i + w2][j + w2][k + w2] == 2 ? 1 : 0); 

    state[i][j + w2][k + w2] = faceRule[idx1][idx2];

    if (state[i][j + w2][k + w2] > 0 && Math.random() > flipP) {
        state[i][j + w2][k + w2] = 3 - state[i][j + w2][k + w2];
    }
}

function f4(i, j, k, w) {
    f1(i, j, k + w, w);
}

function f5(i, j, k, w) {
    f1(i, j + w, k, w);
}

function f6(i, j, k, w) {
    f1(i + w, j, k, w);
}

function evalFaces(i, j, k, w) {
    f1(i, j, k, w);
    f2(i, j, k, w);
    f3(i, j, k, w);
    f4(i, j, k, w);
    f5(i, j, k, w);
    f6(i, j, k, w);
}

function e1(i, j, k, w) {
    let w2 = Math.floor(w / 2);
    if (i < 0 || j - w2 < 0 || k - w2 < 0 || i + w >= K || j + w2 >= K || k + w2 >= K) return;

    let idx1 =  (state[i][j][k]== 1 ? 1 : 0) + (state[i + w][j][k]== 1 ? 1 : 0) + (state[i + w2][j - w2][k]== 1 ? 1 : 0) +
                (state[i + w2][j + w2][k]== 1 ? 1 : 0) + (state[i + w2][j][k + w2]== 1 ? 1 : 0) + (state[i + w2][j][k - w2]== 1 ? 1 : 0);
    let idx2 =  (state[i][j][k]== 1 ? 1 : 0) + (state[i + w][j][k]== 1 ? 1 : 0) + (state[i + w2][j - w2][k]== 1 ? 1 : 0) +
                (state[i + w2][j + w2][k]== 1 ? 1 : 0) + (state[i + w2][j][k + w2]== 1 ? 1 : 0) + (state[i + w2][j][k - w2]== 1 ? 1 : 0);

    state[i + w2][j][k] = edgeRule[idx1][idx2];

    if (state[i + w2][j][k] > 0 && Math.random() > flipP) {
        state[i + w2][j][k] = 3 - state[i + w2][j][k];
    }
}

function e2(i, j, k, w) {
    e1(i, j+w, k, w);
}

function e3(i, j, k, w) {
    e1(i, j, k + w, w);
}

function e4(i, j, k, w) {
    e1(i, j+w, k + w, w);
}

function e5(i, j, k, w) {
    let w2 = Math.floor(w / 2);
    e1(i - w2, j + w2, k, w);
}

function e6(i, j, k, w) {
    let w2 = Math.floor(w / 2);
    e1(i + w2, j + w2, k, w);
}

function e7(i, j, k, w) {
    let w2 = Math.floor(w / 2);
    e1(i - w2, j + w2, k + w, w);
}

function e8(i, j, k, w) {
    let w2 = Math.floor(w / 2);
    e1(i + w2, j + w2, k + w, w);
}

function evalEdges(i, j, k, w) {
    e1(i, j, k, w);
    e2(i, j, k, w);
    e3(i, j, k, w);
    e4(i, j, k, w);
    e5(i, j, k, w);
    e6(i, j, k, w);
    e7(i, j, k, w);
    e8(i, j, k, w);
}

function randomRule(lambda, rng) {
    for (var i = 0; i < 9; i++) {
        for (var j = 0; j < 9 - i; j++) {
            if ((i == 0 && j == 0) || rng() > lambda) cubeRule[i][j] = 0;
            else {
                cubeRule[i][j] = Math.floor(rng() * 2.) + 1;
            }
        }
    }

    for (var i = 0; i < 7; i++) {
        for (var j = 0; j < 7 - i; j++) {
            if ((i == 0 && j == 0) || rng() > lambda) faceRule[i][j] = 0;
            else {
                faceRule[i][j] = Math.floor(rng() * 2.) + 1;
            }
        }
    }

    for (var i = 0; i < 7; i++) {
        for (var j = 0; j < 7 - i; j++) {
            if ((i == 0 && j == 0) || rng() > lambda) edgeRule[i][j] = 0;
            else {
                edgeRule[i][j] = Math.floor(rng() * 2.) + 1;
            }
        }
    }
}

function initState() {
    for (var i = 0; i < K; i++) {
        for (var j = 0; j < K; j++) {
            state[i][j][0] = Math.floor(Math.random() * 2) + 1;
        }
    }
}

async function evalState() {
    for (var w = K - 1; w >= 2; w = Math.floor(w / 2)) {
        document.querySelector("#generation-stage").innerHTML = `resolution ${2 * (K - 1) / w}`;
        await new Promise(r => setTimeout(r, 10));
        for (var i = 0; i < K - 1; i = i + w) {
            for (var j = 0; j < K - 1; j = j + w) {
                for (var k = 0; k < K - 1; k = k + w) {
                    evalCube(i, j, k, w);
                }
            }
        }
        for (var i = 0; i < K - 1; i = i + w) {
            for (var j = 0; j < K - 1; j = j + w) {
                for (var k = 0; k < K - 1; k = k + w) {
                    evalFaces(i, j, k, w);
                }
            }
        }
        for (var i = 0; i < K - 1; i = i + w) {
            for (var j = 0; j < K - 1; j = j + w) {
                for (var k = 0; k < K - 1; k = k + w) {
                    evalEdges(i, j, k, w);
                }
            }
        }  
    }
}