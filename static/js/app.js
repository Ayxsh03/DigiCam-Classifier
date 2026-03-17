/* ═══════════════════════════════════════════════════════════════════════════════
   DigiCam — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
    // ─── Tab Switching ───────────────────────────────────────────────────────────
    const tabs = document.querySelectorAll(".tab");
    const sections = document.querySelectorAll(".section");

    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((t) => t.classList.remove("active"));
            sections.forEach((s) => s.classList.remove("active"));
            tab.classList.add("active");
            document.getElementById(tab.dataset.target).classList.add("active");
        });
    });

    // ─── Canvas Drawing ──────────────────────────────────────────────────────────
    const canvas = document.getElementById("draw-canvas");
    const ctx = canvas.getContext("2d");
    let drawing = false;

    // Fill black background
    function clearCanvas() {
        ctx.fillStyle = "#111118";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    clearCanvas();

    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#fff";

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        if (e.touches) {
            return {
                x: (e.touches[0].clientX - rect.left) * scaleX,
                y: (e.touches[0].clientY - rect.top) * scaleY,
            };
        }
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    function startDraw(e) {
        e.preventDefault();
        drawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!drawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDraw() {
        drawing = false;
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mouseleave", stopDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDraw);

    // ─── Clear ───────────────────────────────────────────────────────────────────
    document.getElementById("btn-clear").addEventListener("click", () => {
        clearCanvas();
        document.getElementById("classify-result").style.display = "none";
    });

    // ─── Loading Overlay ─────────────────────────────────────────────────────────
    const loading = document.getElementById("loading");
    function showLoading() { loading.style.display = "flex"; }
    function hideLoading() { loading.style.display = "none"; }

    // ─── Classify ────────────────────────────────────────────────────────────────
    document.getElementById("btn-classify").addEventListener("click", async () => {
        const dataURL = canvas.toDataURL("image/png");
        showLoading();
        try {
            const res = await fetch("/api/classify", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataURL }),
            });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            renderPrediction(data);
        } catch (err) {
            alert("Classification failed: " + err.message);
        } finally {
            hideLoading();
        }
    });

    // Upload classify
    document.getElementById("upload-input").addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append("image", file);
        showLoading();
        try {
            const res = await fetch("/api/classify", { method: "POST", body: formData });
            const data = await res.json();
            if (data.error) throw new Error(data.error);
            renderPrediction(data);
        } catch (err) {
            alert("Classification failed: " + err.message);
        } finally {
            hideLoading();
            e.target.value = "";
        }
    });

    function renderPrediction(data) {
        const resultCard = document.getElementById("classify-result");
        resultCard.style.display = "block";

        document.getElementById("predicted-digit").textContent = data.digit;
        document.getElementById("confidence-label").textContent =
            (data.confidence * 100).toFixed(1) + "%";

        const barsEl = document.getElementById("prob-bars");
        barsEl.innerHTML = "";

        const maxProb = Math.max(...data.probabilities);

        data.probabilities.forEach((p, i) => {
            const isTop = p === maxProb;
            barsEl.innerHTML += `
                <span class="prob-label">${i}</span>
                <div class="prob-track">
                    <div class="prob-fill${isTop ? " top" : ""}" style="width: ${(p * 100).toFixed(1)}%"></div>
                </div>
                <span class="prob-value">${(p * 100).toFixed(1)}%</span>
            `;
        });
    }

    // ─── Digit Pickers ───────────────────────────────────────────────────────────
    let digitA = 3;
    let digitB = 8;

    function setupPicker(pickerId, setter) {
        const grid = document.getElementById(pickerId);
        grid.querySelectorAll(".digit-btn").forEach((btn) => {
            btn.addEventListener("click", () => {
                grid.querySelectorAll(".digit-btn").forEach((b) => b.classList.remove("selected"));
                btn.classList.add("selected");
                setter(parseInt(btn.dataset.digit));
            });
        });
    }

    setupPicker("picker-a", (v) => { digitA = v; });
    setupPicker("picker-b", (v) => { digitB = v; });

    // ─── Steps Slider ────────────────────────────────────────────────────────────
    const stepsSlider = document.getElementById("steps-slider");
    const stepsVal = document.getElementById("steps-val");
    stepsSlider.addEventListener("input", () => {
        stepsVal.textContent = stepsSlider.value;
    });

    // ─── Interpolate ─────────────────────────────────────────────────────────────
    document.getElementById("btn-interpolate").addEventListener("click", async () => {
        const steps = parseInt(stepsSlider.value);
        showLoading();
        try {
            const res = await fetch("/api/interpolate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ digit_a: digitA, digit_b: digitB, steps }),
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.error || "Server error");
            }

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);

            const resultCard = document.getElementById("interpolate-result");
            resultCard.style.display = "block";

            const gifImg = document.getElementById("morph-gif");
            gifImg.src = url;

            const dlLink = document.getElementById("download-gif");
            dlLink.href = url;
            dlLink.download = `morph_${digitA}_to_${digitB}.gif`;
        } catch (err) {
            alert("Interpolation failed: " + err.message);
        } finally {
            hideLoading();
        }
    });
});
