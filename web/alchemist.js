import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { repairWidgets } from "./widget_repair.js";

const MAIN = "AlchemistPrompt";
const BATCH = "AlchemistDatasetCaptioner";
const NODES = new Set([
  MAIN,
  BATCH,
  "AlchemistKrea2",
  "AlchemistLoraCaption",
  "AlchemistH3",
]);

// Accent per node so they are distinguishable at a glance on a busy graph.
const NODE_ACCENT = {
  AlchemistKrea2: "#ffe27a",
  AlchemistLoraCaption: "#7bff9e",
  AlchemistH3: "#5ef2ff",
  AlchemistDatasetCaptioner: "#7bff9e",
  AlchemistPrompt: "#7b5cff",
};

const NODE_TITLE = {
  AlchemistKrea2: "KREA 2 PROMPT",
  AlchemistLoraCaption: "KREA 2 LORA CAPTION",
  AlchemistH3: "MINIMAX H3 DOC",
  AlchemistDatasetCaptioner: "DATASET CAPTIONER",
  AlchemistPrompt: "ANY ENGINE",
};

let PROVIDERS = [];

const ENGINE_COLOR = {
  krea2: "#ffe27a",
  krea2_lora: "#7bff9e",
  h3: "#5ef2ff",
};

function toast(html, ms = 5200) {
  let el = document.querySelector(".alchemist-toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "alchemist-toast";
    el.style.cssText =
      "position:fixed;right:18px;bottom:18px;z-index:99999;max-width:460px;" +
      "padding:11px 13px;background:#0d0a14;color:#eaf6ff;border:1px solid #7b5cff;" +
      "border-radius:8px;font:12px/1.45 ui-sans-serif,system-ui;" +
      "box-shadow:0 8px 28px rgba(0,0,0,.55);opacity:1;transition:opacity .3s";
    document.body.appendChild(el);
  }
  el.innerHTML = html;
  el.style.opacity = "1";
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    el.style.opacity = "0";
  }, ms);
}

function widgetOf(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

function val(node, name, fallback = "") {
  const w = widgetOf(node, name);
  if (!w) return fallback;
  return w.value ?? fallback;
}

function providerByLabel(label) {
  return PROVIDERS.find((p) => p.label === label) || PROVIDERS.find((p) => p.key === label);
}

function currentProviderKey(node) {
  const found = providerByLabel(String(val(node, "provider", "")));
  return found ? found.key : "xai";
}

async function loadProviders() {
  if (PROVIDERS.length) return PROVIDERS;
  try {
    const res = await api.fetchApi("/alchemist/providers");
    PROVIDERS = await res.json();
  } catch (_) {
    PROVIDERS = [];
  }
  return PROVIDERS;
}

function setComboOptions(widget, values, keepIfPresent = true) {
  if (!widget || !Array.isArray(values) || !values.length) return;
  widget.options = widget.options || {};
  widget.options.values = values;
  if (!keepIfPresent || !values.includes(widget.value)) widget.value = values[0];
}

async function fetchModels(node, quiet = false) {
  const key = currentProviderKey(node);
  const prov = PROVIDERS.find((p) => p.key === key);
  const qs = new URLSearchParams({
    provider: key,
    key: String(val(node, "api_key", "")),
    base_url: String(val(node, "custom_base_url", "")),
  });
  try {
    const res = await api.fetchApi(`/alchemist/models?${qs}`);
    const data = await res.json();
    const models = data.models || [];
    setComboOptions(widgetOf(node, "model"), models);
    node.setDirtyCanvas(true, true);
    if (!quiet) {
      toast(
        `<b>${prov?.label || key}</b><br>${models.length} models.<br>` +
          `<span style="opacity:.7">${models.slice(0, 4).join(" · ")}</span>`,
      );
    }
    return models;
  } catch (err) {
    if (!quiet) toast(`<b>MODEL FETCH FAILED</b><br>${err.message || err}`);
    return [];
  }
}

async function vaultKey(node) {
  const key = String(val(node, "api_key", "")).trim();
  const provKey = currentProviderKey(node);
  if (!key) {
    toast("<b>VAULT EMPTY</b><br>Paste a key in api_key first.");
    return;
  }
  try {
    const res = await api.fetchApi("/alchemist/vault", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: provKey,
        api_key: key,
        base_url: String(val(node, "custom_base_url", "")),
      }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "vault failed");
    setComboOptions(widgetOf(node, "model"), data.models || []);
    PROVIDERS = [];
    await loadProviders();
    toast(`<b>KEY VAULTED</b><br>${provKey} · ${data.count} models on the table.`);
    node.setDirtyCanvas(true, true);
  } catch (err) {
    toast(`<b>VAULT FUMBLE</b><br>${err.message || err}`);
  }
}

async function preview(node) {
  const body = {
    provider: currentProviderKey(node),
    model: val(node, "model", ""),
    target_engine: val(node, "target_engine", "krea2"),
    caption_mode: val(node, "caption_mode", "regular"),
    simple_prompt: val(node, "simple_prompt", ""),
    extra_direction: val(node, "extra_direction", ""),
    edit_instruction: val(node, "edit_instruction", ""),
    controlnet_lock: val(node, "controlnet_lock", "auto"),
    lora_kind: val(node, "lora_kind", "character"),
    trigger_word: val(node, "trigger_word", ""),
    caption_words: Number(val(node, "caption_words", 45)),
    omit_terms: val(node, "omit_terms", ""),
    nsfw: Boolean(val(node, "nsfw", true)),
    temperature: Number(val(node, "temperature", 0.8)),
    max_tokens: Number(val(node, "max_tokens", 1400)),
    clip_seconds: Number(val(node, "clip_seconds", 8)),
    h3_aspect: val(node, "h3_aspect", "16:9"),
    h3_music: val(node, "h3_music", "auto"),
    api_key: val(node, "api_key", ""),
    custom_base_url: val(node, "custom_base_url", ""),
    reasoning: val(node, "reasoning", "low"),
  };
  toast("<b>ALCHEMY IN FLIGHT</b><br>Text-only preview, no queue…");
  try {
    const res = await api.fetchApi("/alchemist/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "preview failed");
    node.alcStatus = `PREVIEW • ${data.model} • ${data.prompt.length} chars`;
    toast(
      `<b>PREVIEW · ${data.model}</b><br><span style="opacity:.85">` +
        `${data.prompt.slice(0, 420).replace(/</g, "&lt;")}…</span>`,
      9000,
    );
    node.setDirtyCanvas(true, true);
  } catch (err) {
    toast(`<b>ALCHEMY SPILLED</b><br>${err.message || err}`);
  }
}

function passwordize(node) {
  const w = widgetOf(node, "api_key");
  if (!w) return;
  const apply = () => {
    const el = w.inputEl || w.element || w.input_element;
    if (el && el.tagName === "INPUT") {
      el.type = "password";
      el.autocomplete = "off";
      el.spellcheck = false;
    }
  };
  apply();
  setTimeout(apply, 60);
  setTimeout(apply, 400);
}

function repairWidgetValues(node) {
  const fixed = repairWidgets(node.widgets, schemaSpecs(node));
  if (fixed.length) {
    toast(
      `<b>REPAIRED ${fixed.length} WIDGET${fixed.length > 1 ? "S" : ""}</b><br>` +
        `<span style="opacity:.8">${fixed.join(", ")}</span><br>` +
        "This workflow was saved by a build that mis-ordered widgets. Re-save it.",
      9000,
    );
    node.setDirtyCanvas(true, true);
  }
  return fixed.length;
}

function hookCombos(node) {
  const bind = (name, fn) => {
    const w = widgetOf(node, name);
    if (!w || w._alcHooked) return;
    w._alcHooked = true;
    const prev = w.callback;
    w.callback = function (v) {
      try {
        prev?.apply(this, arguments);
      } catch (_) {
        /* upstream callback is optional */
      }
      fn(v ?? w.value);
    };
  };

  bind("provider", async () => {
    await loadProviders();
    const prov = PROVIDERS.find((p) => p.key === currentProviderKey(node));
    await fetchModels(node, true);
    if (prov) {
      const ready = prov.ready
        ? '<span style="color:#7bff9e">key ready</span>'
        : '<span style="color:#ff7b7b">no key — paste one and VAULT</span>';
      toast(
        `<b>${prov.label}</b><br>${ready}` +
          `${prov.vision ? "" : "<br>no vision on this provider"}` +
          `${prov.notes ? `<br><span style="opacity:.65">${prov.notes}</span>` : ""}`,
      );
    }
  });
  bind("target_engine", () => node.setDirtyCanvas(true, true));
  bind("action", (v) => {
    const choice = String(v ?? "").toLowerCase();
    const reset = () => {
      const w = widgetOf(node, "action");
      if (w) w.value = "idle";
      node.setDirtyCanvas(true, true);
    };
    if (choice.includes("vault")) {
      reset();
      vaultKey(node);
    } else if (choice.includes("fetch")) {
      reset();
      fetchModels(node);
    } else if (choice.includes("preview")) {
      reset();
      preview(node);
    }
  });
}

function drawChrome(node, ctx) {
  const w = node.size[0];
  const h = node.size[1];
  const cls = node.comfyClass;
  const accent = NODE_ACCENT[cls] || "#7b5cff";
  const prov = PROVIDERS.find((p) => p.key === currentProviderKey(node));
  const nsfw = Boolean(val(node, "nsfw", true));

  ctx.save();

  // Header band: a thin accent wash rather than the old animated gradient, so a
  // graph full of these nodes stays readable.
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, accent + "33");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, 18);
  ctx.fillStyle = accent;
  ctx.fillRect(0, 17, w, 1);

  let title = NODE_TITLE[cls] || "ALCHEMIST";
  if (cls === MAIN) {
    const engine = String(val(node, "target_engine", "krea2"));
    title = { krea2: "KREA 2", krea2_lora: "LORA CAPTION", h3: "MINIMAX H3" }[engine] || title;
  }
  ctx.font = "700 9px Orbitron, Syne, ui-sans-serif";
  ctx.textAlign = "left";
  ctx.fillStyle = accent;
  ctx.fillText(title, 10, 12);

  // Right side carries only what can actually bite you: missing key, no vision.
  const bits = [];
  if (prov) {
    bits.push(prov.label.split(" ")[0].toUpperCase());
    if (!prov.ready) bits.push("NO KEY");
    else if (!prov.vision) bits.push("NO VISION");
  }
  bits.push(nsfw ? "NSFW" : "SFW");
  ctx.textAlign = "right";
  ctx.fillStyle = prov && !prov.ready ? "#ff7b7b" : "rgba(234,246,255,0.55)";
  ctx.fillText(bits.join(" · "), w - 10, 12);

  if (node.alcStatus) {
    ctx.font = "8px ui-sans-serif, system-ui";
    ctx.fillStyle = "rgba(234,246,255,0.4)";
    ctx.textAlign = "center";
    ctx.fillText(String(node.alcStatus).slice(0, 78), w / 2, h - 6);
  }
  ctx.restore();
}

async function decorate(node) {
  if (!node || !NODES.has(node.comfyClass) || node._alcDecorated) return;
  node._alcDecorated = true;
  const accent = NODE_ACCENT[node.comfyClass] || "#7b5cff";
  node.color = "#171029";
  node.bgcolor = "#0b0812";
  node.boxcolor = accent;
  try {
    // The focused nodes are short now; only the two big ones need the height.
    const tall = node.comfyClass === MAIN || node.comfyClass === BATCH;
    node.setSize([
      Math.max(node.size?.[0] || 0, 420),
      Math.max(node.size?.[1] || 0, tall ? 520 : 360),
    ]);
  } catch (_) {
    /* size is optional */
  }
  await loadProviders();
  hookCombos(node);
  passwordize(node);
  repairWidgetValues(node);
  setTimeout(() => {
    hookCombos(node);
    passwordize(node);
    repairWidgetValues(node);
      fetchModels(node, true);
  }, 120);
}

function decorateAll() {
  for (const node of app.graph?._nodes || app.graph?.nodes || []) decorate(node);
}

app.registerExtension({
  name: "alchemist.any.llm",
  async setup() {
    await loadProviders();
    decorateAll();
    setTimeout(decorateAll, 900);
  },
  async afterConfigureGraph() {
    decorateAll();
    // Saved values land after decoration, so repair once more on load.
    for (const node of app.graph?._nodes || []) {
      if (NODES.has(node.comfyClass)) {
        repairWidgetValues(node);
            }
    }
  },
  nodeCreated(node) {
    decorate(node);
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.has(nodeData?.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      decorate(this);
      return r;
    };

    const onDraw = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDraw?.apply(this, arguments);
      try {
        drawChrome(this, ctx);
      } catch (_) {
        /* chrome is ornamental */
      }
      return r;
    };

    const getExtra = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (_, options) {
      getExtra?.apply(this, arguments);
      options?.unshift(
        { content: "🔑 Vault API key", callback: () => vaultKey(this) },
        { content: "↻ Fetch models", callback: () => fetchModels(this) },
        { content: "✧ Preview (no queue)", callback: () => preview(this) },
        null,
      );
    };
  },
});
