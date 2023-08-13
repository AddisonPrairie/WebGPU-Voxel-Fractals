function focalMap(a) {
    return Math.exp((a - .2) * 7.);
}

function initUI(flags, callbacks, size) {
    uiBindSlider(
        document.querySelector("#focal-distance-slider"),
        focalMap,
        document.querySelector("#focal-distance-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#aperture-slider"),
        a => a,
        document.querySelector("#aperture-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#sun-brightness-slider"),
        a => a,
        document.querySelector("#sun-brightness-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#sky-brightness-slider"),
        a => a,
        document.querySelector("#sky-brightness-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#azimuth-slider"),
        a => a,
        document.querySelector("#azimuth-value"),
        4,
        flags
    );
    uiBindSlider(
        document.querySelector("#zenith-slider"),
        a => a,
        document.querySelector("#zenith-value"),
        4,
        flags
    );

    const dummyFlags = {};
    uiBindSlider(
        document.querySelector("#speed-slider"),
        a => a,
        document.querySelector("#speed-value"),
        4,
        dummyFlags
    )

    document.querySelector("#sky-color").oninput = () => {flags["changed"] = true;};
    document.querySelector("#sun-color").oninput = () => {flags["changed"] = true;};

    //prng from: https://www.pcg-random.org/posts/bob-jenkins-small-prng-passes-practrand.html
    function jsf32(a, b, c, d) {
        return function() {
            a |= 0; b |= 0; c |= 0; d |= 0;
            var t = a - (b << 27 | b >>> 5) | 0;
            a = b ^ (c << 17 | c >>> 15);
            b = c + d | 0;
            c = d + t | 0;
            d = a + t | 0;
            return (d >>> 0) / 4294967296;
        }
    }

    let rng = null;
    document.querySelector("#rule-seed").oninput = () =>  {
        const str = document.querySelector("#rule-seed").value;
        const vals= str.split("-");
        for (var i = 0; i < vals.length; i++) {
            vals[i] = parseInt(vals[i]);
            if (isNaN(vals[i])) {vals[i] = 0;}
        }
        rng = jsf32(vals[0], vals[1], vals[2], vals[3]);
        newRule(rng);
    };

    document.querySelector("#new-rule").onclick = () => {
        let valstr = "";
        for (var i = 0; i < 4; i++) {
            valstr += `${Math.floor(Math.random() * 1000)}`;
            if (i != 3) {valstr += "-"};
        }
        document.querySelector("#rule-seed").value = valstr;
        document.querySelector("#rule-seed").oninput();
    }

    document.querySelector("#reset-render").onclick = () => {
        flags["changed"] = true;
    };

    document.querySelector("#regenerate-scene").onclick = async () => {
        flags["lock-input"] = true;
        await newStateRandom();
        document.querySelector("#generation-stage").innerHTML = `rebuilding AS...`;
        await new Promise(r => setTimeout(r, 1));
        await callbacks["setScene"]();
        document.querySelector("#generation-stage").innerHTML = `complete`;
        flags["lock-input"] = false;
    };

    const tonemap = document.querySelector("#tone-mapping");
    const aces = document.querySelector("#aces");
    const reinhard = document.querySelector("#reinhard");

    const setACES = () => {
        aces.checked = true;
        reinhard.checked = false;
        tonemap.dataset.mode = 0;
    };
    const setReinhard = () => {
        aces.checked = false;
        reinhard.checked = true;
        tonemap.dataset.mode = 1;
    }

    aces.oninput = () => {
        console.log(tonemap.dataset.mode);
        if (tonemap.dataset.mode == 0) {return;}
        setACES();
        flags["changed"] = true;
    };
    reinhard.oninput = () => {
        if (tonemap.dataset.mode == 1) {return;}
        setReinhard();
        flags["changed"] = true;
    };

    setReinhard();

    const helpWindow = document.querySelector(".help");
    const helpBackground = document.querySelector(".help-background");
    document.querySelector("#help").onclick = () => {
        helpWindow.style.display = "";
        helpBackground.style.display = "";
    };
    helpBackground.onclick = () => {
        helpWindow.style.display = "none";
        helpBackground.style.display = "none";
    };
    helpBackground.onclick();

    document.querySelector("#copy-seed").onclick = () => {
        const root = window.location.href.split("?")[0];

        navigator.clipboard.writeText(root + `?size=${size}&seed=${document.querySelector("#rule-seed").value}`);
    };
}

//writes all UI values to a typed array to be sent to the GPU
function getUIValues() {
    const returned = new Float32Array(4 + 4 + 4 + 4);
    const suncol = hexToRGB(document.querySelector("#sun-color").value);
    const skycol = hexToRGB(document.querySelector("#sky-color").value);
    returned[0] = focalMap(parseFloat(document.querySelector("#focal-distance-slider").value));
    returned[1] = parseFloat(document.querySelector("#aperture-slider").value);
    returned[2] = 3.1415;
    returned[3] = 3.1415;
    returned[4] = suncol[0]; returned[5] = suncol[1]; returned[6] = suncol[2];
    returned[7] = parseFloat(document.querySelector("#sun-brightness-slider").value);
    returned[8] = skycol[0]; returned[9] = skycol[1]; returned[10] = skycol[2];
    returned[11]= parseFloat(document.querySelector("#sky-brightness-slider").value);
    returned[12]=parseFloat(document.querySelector("#azimuth-slider").value * Math.PI / 180.);
    returned[13]=parseFloat(Math.PI / 2. + document.querySelector("#zenith-slider").value * Math.PI / 180.);
    returned[14]=document.querySelector("#tone-mapping").dataset.mode;
    return returned;
}

function hexToRGB(hex) {
    return [
        parseInt(hex[1] + hex[2], 16) / 255.,
        parseInt(hex[3] + hex[4], 16) / 255.,
        parseInt(hex[5] + hex[6], 16) / 255.
    ]
}

function uiBindSlider(slider, sliderCurve, text, maxChars, flags) {
    slider.oninput = () => {
        flags["changed"] = true;
        const val = sliderCurve(slider.value);
        let strVal = "" + val;
        let outstr = "";
        for (var x = 0; x < maxChars && x < strVal.length; x++) {
            outstr += strVal[x];
        }
        if (outstr.length < maxChars) {
            if (Math.floor(val) == val) {
                outstr += ".";
            }
            while(outstr.length < maxChars) {
                outstr += "0";
            }
        }
        text.innerHTML = outstr;
    }
    slider.oninput();
}