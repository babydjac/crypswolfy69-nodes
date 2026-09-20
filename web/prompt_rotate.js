import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAMES = new Set(["PromptRotate", "PromptRotatePick"]);

function toast(html) {
  let el = document.querySelector(".prompt-rotate-toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "prompt-rotate-toast";
    el.style.cssText =
      "position:fixed;right:18px;bottom:18px;z-index:99999;max-width:420px;" +
      "padding:10px 12px;background:#120c14;color:#ffe9fb;border:1px solid #ff2bd6;" +
      "border-radius:8px;font:12px/1.4 ui-sans-serif,system-ui;opacity:1;transition:opacity .3s";
    document.body.appendChild(el);
  }
  el.innerHTML = html;
  el.style.opacity = "1";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.style.opacity = "0";
  }, 5000);
}

function widgetOf(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function widgetValue(node, name, fallback = "") {
  const w = widgetOf(node, name);
  if (!w) return fallback;
  return w.value ?? fallback;
}

function isRotateNode(node) {
  const names = [
    node?.comfyClass,
    node?.constructor?.comfyClass,
    node?.type,
    node?.constructor?.type,
    node?.constructor?.nodeData?.name,
  ].map((x) => String(x || ""));
  return names.some((n) => NODE_NAMES.has(n)) || names.some((n) => n.includes("PROMPT ROTATE"));
}

function parsePrompts(text, splitMode) {
  const raw = String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .trim();
  if (!raw) return [];

  const clean = (parts) => parts.map((p) => String(p || "").trim()).filter(Boolean);
  const mode = String(splitMode || "auto").toLowerCase();
  const splitNumbered = (s) => clean(s.split(/^[ \t]*\d+[.)][ \t]+/m));
  const splitBlank = (s) => clean(s.split(/\n\s*\n/));
  const splitLines = (s) => clean(s.split("\n"));
  const splitRule = (s) => clean(s.split(/^[ \t]*(?:-{3,}|={3,})[ \t]*$/m));

  if (mode === "numbered") return splitNumbered(raw);
  if (mode === "blank lines") return splitBlank(raw);
  if (mode === "one per line") return splitLines(raw);

  const numbered = splitNumbered(raw);
  if (numbered.length >= 2) return numbered;
  const ruled = splitRule(raw);
  if (ruled.length >= 2) return ruled;
  const blanks = splitBlank(raw);
  if (blanks.length >= 2) return blanks;
  const lines = splitLines(raw);
  if (lines.length >= 2 && lines.every((line) => line.length > 24)) return lines;
  return [raw];
}

function growTextarea(node) {
  const w = widgetOf(node, "prompts");
  const el = w?.inputEl || w?.element || w?.input_element;
  if (el && el.tagName === "TEXTAREA") {
    el.rows = 10;
    el.style.minHeight = "140px";
    el.style.font = "11px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  }
}

function countPrompts(node) {
  const items = parsePrompts(widgetValue(node, "prompts"), widgetValue(node, "split_mode", "auto"));
  toast(`<b>PROMPT ROTATE</b><br>${items.length} prompts parsed.`);
  return items;
}

function bumpSeedsInOutput(output, offset) {
  if (!output || !offset) return;
  for (const graphNode of Object.values(output)) {
    const inputs = graphNode?.inputs;
    if (!inputs || typeof inputs !== "object") continue;
    for (const key of Object.keys(inputs)) {
      if (!/^(seed|noise_seed)$/i.test(key)) continue;
      if (typeof inputs[key] === "number") inputs[key] = inputs[key] + offset;
    }
  }
}

function outputNode(output, node) {
  if (!output) return null;
  return output[String(node.id)] || output[node.id] || null;
}

async function graphSnapshot() {
  if (typeof app.graphToPrompt !== "function") {
    throw new Error("app.graphToPrompt is missing — hard-refresh Comfy");
  }
  const raw = await app.graphToPrompt();
  return {
    output: raw.output || raw,
    workflow: raw.workflow || app.graph?.serialize?.() || {},
  };
}

async function queueAll(node) {
  const items = parsePrompts(widgetValue(node, "prompts"), widgetValue(node, "split_mode", "auto"));
  if (!items.length) {
    toast("<b>PROMPT ROTATE</b><br>Nothing to queue. Paste prompts first.");
    return;
  }

  const bump = Boolean(widgetValue(node, "bump_graph_seeds", true));
  toast(`<b>PROMPT ROTATE</b><br>Queueing ${items.length} jobs…`);

  let snap;
  try {
    snap = await graphSnapshot();
  } catch (err) {
    toast(`<b>QUEUE FAILED</b><br>${err.message || err}`);
    return;
  }

  if (!outputNode(snap.output, node)) {
    toast("<b>QUEUE FAILED</b><br>Wire this node into the graph (prompt → Krea text).");
    return;
  }

  let queued = 0;
  try {
    for (let i = 0; i < items.length; i++) {
      const output = structuredClone(snap.output);
      const inputs = outputNode(output, node).inputs;
      inputs.prompt_index = i + 1;
      inputs.forced_prompt = items[i];
      inputs.run = "idle";
      if (bump) bumpSeedsInOutput(output, i);
      await api.queuePrompt(0, { output, workflow: snap.workflow });
      queued += 1;
    }
    toast(`<b>QUEUED ${queued}</b><br>One job per prompt, in order.`);
  } catch (err) {
    toast(`<b>STOPPED AT ${queued}/${items.length}</b><br>${err.message || err}`);
  }
}

function resetRun(node) {
  const w = widgetOf(node, "run");
  if (w) w.value = "idle";
}

function hookRunCombo(node) {
  const w = widgetOf(node, "run");
  if (!w || w._prHooked) return;
  w._prHooked = true;
  const prev = w.callback;
  w.callback = function (v) {
    try {
      prev?.apply(this, arguments);
    } catch (_) {
      /* ignore upstream callback errors */
    }
    const val = String(v ?? w.value ?? "").toLowerCase();
    if (val.includes("queue")) {
      resetRun(node);
      queueAll(node);
    } else if (val.includes("count")) {
      resetRun(node);
      countPrompts(node);
    }
  };
}

function decorate(node) {
  if (!node || !isRotateNode(node) || node._promptRotateDecorated) return;
  node._promptRotateDecorated = true;
  node.color = "#2a1230";
  node.bgcolor = "#100814";
  try {
    node.setSize([Math.max(node.size?.[0] || 0, 480), Math.max(node.size?.[1] || 0, 520)]);
  } catch (_) {
    /* size is optional */
  }
  hookRunCombo(node);
  setTimeout(() => {
    hookRunCombo(node);
    growTextarea(node);
  }, 60);
  setTimeout(() => growTextarea(node), 400);
}

function decorateAll() {
  for (const node of app.graph?._nodes || app.graph?.nodes || []) decorate(node);
}

app.registerExtension({
  name: "prompt.rotate.queue",
  async setup() {
    decorateAll();
    setTimeout(decorateAll, 800);
  },
  async afterConfigureGraph() {
    decorateAll();
  },
  nodeCreated(node) {
    decorate(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const name = nodeData?.name || nodeData?.comfyClass || "";
    if (!NODE_NAMES.has(name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      decorate(this);
      return r;
    };

    if (name !== "PromptRotate") return;
    const getExtra = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      getExtra?.apply(this, arguments);
      options?.unshift(
        { content: "▶ Queue All Prompts", callback: () => queueAll(this) },
        { content: "Σ Count Prompts", callback: () => countPrompts(this) },
        null,
      );
    };
  },
});
