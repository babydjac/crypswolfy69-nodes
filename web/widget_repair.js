/** Repair widget values that no longer match the server schema.
 *
 * Workflows saved by a build that reordered `node.widgets` come back with every
 * value shifted one slot: a combo holding a boolean, a number holding "". Comfy
 * then rejects the prompt with validation errors the user cannot act on, so the
 * values are coerced back to something legal instead.
 *
 * Kept free of any ComfyUI import so it can be unit tested directly with node.
 */

export function comboOptions(spec) {
  if (!spec) return null;
  if (Array.isArray(spec[0])) return spec[0];
  const opts = (spec[1] || {}).options;
  return Array.isArray(opts) ? opts : null;
}

export function specKind(spec) {
  if (!spec) return "";
  return Array.isArray(spec[0]) ? "COMBO" : String(spec[0]);
}

/**
 * @param {Array<{name: string, value: unknown}>} widgets mutated in place
 * @param {Record<string, unknown[]>} specs server schema, name -> [type, config]
 * @returns {string[]} names that were changed
 */
export function repairWidgets(widgets, specs) {
  const fixed = [];

  for (const w of widgets || []) {
    const spec = specs?.[w.name];
    if (!spec) continue;
    const config = spec[1] || {};
    const fallback = config.default;
    const options = comboOptions(spec);

    if (options) {
      if (!options.includes(w.value)) {
        w.value = options.includes(fallback) ? fallback : options[0];
        fixed.push(w.name);
      }
      continue;
    }

    const kind = specKind(spec);

    if (kind === "INT" || kind === "FLOAT") {
      const wasBool = typeof w.value === "boolean";
      let num = typeof w.value === "number" ? w.value : parseFloat(w.value);
      if (!Number.isFinite(num) || wasBool) {
        num = typeof fallback === "number" ? fallback : 0;
        fixed.push(w.name);
      }
      if (typeof config.min === "number" && num < config.min) {
        num = typeof fallback === "number" ? fallback : config.min;
        fixed.push(w.name);
      }
      if (typeof config.max === "number" && num > config.max) {
        num = typeof fallback === "number" ? fallback : config.max;
        fixed.push(w.name);
      }
      w.value = kind === "INT" ? Math.round(num) : num;
      continue;
    }

    if (kind === "BOOLEAN" && typeof w.value !== "boolean") {
      w.value = typeof fallback === "boolean" ? fallback : true;
      fixed.push(w.name);
      continue;
    }

    if (kind === "STRING" && typeof w.value !== "string") {
      w.value = typeof fallback === "string" ? fallback : "";
      fixed.push(w.name);
    }
  }

  return [...new Set(fixed)];
}
