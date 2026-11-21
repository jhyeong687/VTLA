const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

const contentsLinks = [...document.querySelectorAll(".contents a")];
const chapters = contentsLinks.map((link) => document.querySelector(link.hash));
const progress = document.querySelector(".reading-progress span");
const masthead = document.querySelector(".masthead");
let scrollFrame = 0;

function updateReadingPosition() {
  scrollFrame = 0;
  const root = document.documentElement;
  const distance = root.scrollHeight - root.clientHeight;
  progress.style.transform = `scaleX(${distance > 0 ? scrollY / distance : 0})`;

  const marker = masthead.offsetHeight + Math.min(innerHeight * 0.28, 180);
  let active = 0;
  chapters.forEach((chapter, index) => {
    if (chapter.getBoundingClientRect().top <= marker) active = index;
  });
  if (distance > 0 && scrollY >= distance - 2) active = chapters.length - 1;
  contentsLinks.forEach((link, index) => {
    if (index === active) link.setAttribute("aria-current", "location");
    else link.removeAttribute("aria-current");
  });
}

function requestReadingPosition() {
  if (!scrollFrame) scrollFrame = requestAnimationFrame(updateReadingPosition);
}

addEventListener("scroll", requestReadingPosition, { passive: true });
addEventListener("resize", requestReadingPosition);
addEventListener("load", requestReadingPosition);
document.querySelectorAll("details").forEach((details) => {
  details.addEventListener("toggle", requestReadingPosition);
});
requestReadingPosition();

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver((entries) => {
    for (const { target, isIntersecting } of entries) {
      if (!isIntersecting) continue;
      revealObserver.unobserve(target);
      if (reducedMotion.matches) continue;
      target.animate(
        [{ opacity: .55, transform: "translateY(12px)" }, { opacity: 1, transform: "translateY(0)" }],
        { duration: 440, easing: "cubic-bezier(.22,.78,.33,1)" }
      );
    }
  }, { threshold: .12 });
  document.querySelectorAll(".chapter > h2, .chapter > .instrument, .key-facts, .finding").forEach((element) => {
    revealObserver.observe(element);
  });
}

const architecture = document.querySelector("#architecture");
const variants = {
  baseline: {
    encoder: "Bypassed",
    detail: "Tactile input disabled",
    observe: "The policy receives RGB frames, the task instruction and the robot’s joint state.",
    condition: "PaliGemma builds the shared context. The spatial force map stays outside the policy.",
    route: ["vision"]
  },
  prefix: {
    encoder: "CNN encoder",
    detail: "Contact map to prefix token",
    observe: "Three cameras observe the task and Paxini samples normal force across the gripper’s two contact surfaces.",
    condition: "A CNN converts the force map into a tactile token that joins the PaliGemma prefix.",
    route: ["contact", "token"]
  },
  gated: {
    encoder: "Contact statistics",
    detail: "Local and global force features",
    observe: "Three cameras observe the task and Paxini samples normal force across the gripper’s two contact surfaces.",
    condition: "Local and global contact statistics gate the conditioning supplied directly to the action expert.",
    route: ["contact", "gate"]
  }
};

function setupArchitecture(figure) {
  const flow = figure.querySelector(".flow");
  const svg = flow.querySelector(".flow-connections");
  const variantsInput = figure.querySelector(".variant-select");
  const controls = figure.querySelector(".flow-controls");
  const stepButtons = [...controls.querySelectorAll("[data-step]")];
  const play = controls.querySelector(".flow-play");
  const playLabel = play.querySelector("[data-flow-play-label]");
  const stepNote = figure.querySelector("[data-flow-step]");
  const stages = ["observe", "condition", "predict"];
  const svgNS = "http://www.w3.org/2000/svg";
  let playbackTimer;
  let runningSignals = [];

  // Each edge names its actual endpoints. Mobile changes the ports, not the data flow.
  const routes = {
    vision: { from: ".context", to: ".prefix", desktop: ["right", "left", .36, .36], mobile: ["bottom", "top"] },
    contact: { from: ".touch", to: ".encoder", desktop: ["right", "left"], mobile: ["bottom", "top"], tactile: true },
    token: { from: ".encoder", to: ".prefix", desktop: ["right", "left", .38, .77], mobile: ["left", "right"], tactile: true },
    gate: { from: ".encoder", to: ".expert", desktop: ["right", "left", .55, .55], mobile: ["bottom", "top"], tactile: true },
    expert: { from: ".prefix", to: ".expert", desktop: ["bottom", "top"], mobile: ["bottom", "top"] },
    chunk: { from: ".expert", to: ".action", desktop: ["right", "left"], mobile: ["bottom", "top"] }
  };

  for (const [key, route] of Object.entries(routes)) {
    const group = document.createElementNS(svgNS, "g");
    group.dataset.route = key;
    const line = document.createElementNS(svgNS, "path");
    const signal = document.createElementNS(svgNS, "path");
    line.setAttribute("class", "connection");
    signal.setAttribute("class", "flow-signal");
    signal.setAttribute("pathLength", "1");
    if (route.tactile) {
      group.setAttribute("data-contact", "");
      line.classList.add("tactile");
      signal.classList.add("tactile");
    }
    group.append(line, signal);
    svg.append(group);
    route.line = line;
    route.signal = signal;
  }

  function drawRoutes() {
    const bounds = flow.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const mobile = getComputedStyle(flow).getPropertyValue("--vertical-paths").trim() === "1";
    svg.setAttribute("viewBox", `0 0 ${bounds.width} ${bounds.height}`);

    function port(selector, side, fraction) {
      const rect = flow.querySelector(selector).getBoundingClientRect();
      const left = rect.left - bounds.left;
      const top = rect.top - bounds.top;
      switch (side) {
        case "top": return [left + rect.width * fraction, top];
        case "bottom": return [left + rect.width * fraction, top + rect.height];
        case "left": return [left, top + rect.height * fraction];
        default: return [left + rect.width, top + rect.height * fraction];
      }
    }

    function connectionPath(route) {
      const [startSide, endSide, startPort = .5, endPort = .5] = mobile ? route.mobile : route.desktop;
      const [x1, y1] = port(route.from, startSide, startPort);
      const [targetX, targetY] = port(route.to, endSide, endPort);
      const endGap = 4;
      const start = `M ${x1} ${y1}`;
      // Vertical paths stay centered under their source, including both Gemma inputs.
      if (endSide === "top") return `${start} V ${targetY - endGap}`;

      const x2 = targetX + (endSide === "left" ? -endGap : endGap);
      if (Math.abs(targetY - y1) < .5) return `${start} H ${x2}`;

      // The desktop prefix path bends between columns and enters the card horizontally.
      const midX = (x1 + x2) / 2;
      const bend = Math.min(8, Math.abs(x2 - x1) / 3, Math.abs(targetY - y1) / 2);
      const dx = Math.sign(x2 - x1) * bend;
      const dy = Math.sign(targetY - y1) * bend;
      return [
        start,
        `H ${midX - dx}`,
        `Q ${midX} ${y1} ${midX} ${y1 + dy}`,
        `V ${targetY - dy}`,
        `Q ${midX} ${targetY} ${midX + dx} ${targetY}`,
        `H ${x2}`
      ].join(" ");
    }

    for (const route of Object.values(routes)) {
      const path = connectionPath(route);
      route.line.setAttribute("d", path);
      route.signal.setAttribute("d", path);
    }
  }

  function stopSignals() {
    runningSignals.forEach((animation) => animation.cancel());
    runningSignals = [];
  }

  function selectStage(stage, animate) {
    const variant = variants[figure.dataset.variant];
    flow.dataset.stage = stage;
    stepButtons.forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.step === stage));
    });
    const prediction = "The Gemma action expert uses the context to predict a 50-step action chunk for the six robot joints.";
    stepNote.textContent = stage === "predict" ? prediction : variant[stage];

    stopSignals();
    if (!animate || reducedMotion.matches) return;

    const keys = { observe: ["vision"], condition: variant.route, predict: ["expert", "chunk"] }[stage];
    runningSignals = keys.map((key, index) => routes[key].signal.animate(
      [{ strokeDashoffset: ".13", opacity: 1 }, { strokeDashoffset: "-1", opacity: 1 }],
      { duration: 840, delay: index * 800, easing: "linear" }
    ));
  }

  function stopPlayback() {
    clearTimeout(playbackTimer);
    playbackTimer = undefined;
    play.removeAttribute("data-playing");
    playLabel.textContent = "Play signal flow";
  }

  function playSequence() {
    play.setAttribute("data-playing", "");
    playLabel.textContent = "Stop playback";
    let index = 0;
    function next() {
      const stage = stages[index];
      if (!stage) {
        stopPlayback();
        return;
      }
      selectStage(stage, true);
      index += 1;
      playbackTimer = setTimeout(next, stage === "observe" ? 1500 : 2300);
    }
    next();
  }

  function showVariant(key, animate) {
    const variant = variants[key];
    figure.dataset.variant = key;
    figure.querySelector("[data-encoder-title]").textContent = variant.encoder;
    figure.querySelector("[data-encoder-detail]").textContent = variant.detail;
    drawRoutes();
    selectStage("condition", animate);
  }

  variantsInput.addEventListener("change", (event) => {
    stopPlayback();
    showVariant(event.target.value, true);
  });
  stepButtons.forEach((button) => {
    button.addEventListener("click", () => {
      stopPlayback();
      selectStage(button.dataset.step, true);
    });
  });
  play.addEventListener("click", () => {
    if (play.hasAttribute("data-playing")) {
      stopPlayback();
      stopSignals();
    } else {
      playSequence();
    }
  });
  reducedMotion.addEventListener("change", () => {
    if (reducedMotion.matches) stopSignals();
  });

  const legend = figure.querySelector(".flow-legend");
  [variantsInput, flow, controls, stepNote, legend].forEach((element) => { element.hidden = false; });
  figure.querySelector("[data-flow-fallback]").hidden = true;
  showVariant(variantsInput.querySelector(":checked").value, false);
  if ("ResizeObserver" in window) new ResizeObserver(drawRoutes).observe(flow);
  else addEventListener("resize", drawRoutes);
}
setupArchitecture(architecture);

const metrics = {
  first: { label: "First-action MAE · lower is better", max: 10, ticks: ["0", "5", "10"], lowerIsBetter: true },
  chunk: { label: "Chunk MAE · lower is better", max: 10, ticks: ["0", "5", "10"], lowerIsBetter: true },
  direction: { label: "Direction agreement · higher is better", max: 100, ticks: ["0", "50%", "100%"], lowerIsBetter: false },
  time: { label: "Mean inference time · lower is better", max: .25, ticks: ["0", "0.125 s", "0.25 s"], lowerIsBetter: true }
};
const chart = document.querySelector(".chart");
const results = document.querySelector("#results");
const metricSelect = document.querySelector(".metric-select");
function showMetric(key) {
  const metric = metrics[key];
  const cells = [...results.querySelectorAll(`[data-metric="${key}"]`)];
  const numbers = cells.map((cell) => parseFloat(cell.textContent));
  const best = metric.lowerIsBetter ? Math.min(...numbers) : Math.max(...numbers);
  chart.querySelector(".chart-label").textContent = metric.label;
  cells.forEach((cell, index) => {
    const row = chart.querySelector(`[data-model="${cell.parentElement.dataset.model}"]`);
    row.classList.toggle("featured", numbers[index] === best);
    row.querySelector("output").textContent = cell.textContent;
    row.querySelector(".bar").style.setProperty("--bar-width", `${numbers[index] / metric.max * 100}%`);
  });
  chart.querySelectorAll(".chart-axis span").forEach((tick, index) => {
    tick.textContent = metric.ticks[index];
  });
}
metricSelect.addEventListener("change", (event) => showMetric(event.target.value));
showMetric(metricSelect.querySelector(":checked").value);
metricSelect.hidden = false;

function setupDemo(player) {
  const video = player.querySelector("video");
  if (!video.canPlayType("video/mp4")) return;

  const transport = player.querySelector(".transport");
  const button = transport.querySelector("button");
  const label = button.querySelector(".transport-label");
  const seek = transport.querySelector(".seek");
  const clock = transport.querySelector(".time");

  function timestamp(seconds) {
    const whole = Math.floor(seconds);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }

  function updateTime() {
    const duration = Number(seek.max);
    seek.value = Math.min(video.currentTime, duration);
    clock.value = `${timestamp(Number(seek.value))} / ${timestamp(duration)}`;
  }

  function updateDuration() {
    if (!Number.isFinite(video.duration)) return;
    seek.max = video.duration;
    seek.disabled = false;
    updateTime();
  }

  function updateButton() {
    const playing = !video.paused && !video.ended;
    button.toggleAttribute("data-playing", playing);
    label.textContent = playing ? "Pause" : "Play";
  }

  function restoreNativeControls() {
    transport.hidden = true;
    video.controls = true;
  }

  button.addEventListener("click", async () => {
    if (!video.paused && !video.ended) {
      video.pause();
      return;
    }
    if (video.ended) video.currentTime = 0;
    button.disabled = true;
    try {
      await video.play();
    } catch {
      restoreNativeControls();
    } finally {
      button.disabled = false;
    }
  });

  seek.addEventListener("input", () => {
    video.currentTime = Number(seek.value);
    updateTime();
  });
  video.addEventListener("loadedmetadata", updateDuration);
  video.addEventListener("timeupdate", updateTime);
  video.addEventListener("play", updateButton);
  video.addEventListener("pause", updateButton);
  video.addEventListener("ended", updateButton);
  video.addEventListener("error", restoreNativeControls);
  video.querySelector("source").addEventListener("error", restoreNativeControls);
  video.controls = false;
  transport.hidden = false;
  updateDuration();
}
document.querySelectorAll("#demo .demo").forEach(setupDemo);
