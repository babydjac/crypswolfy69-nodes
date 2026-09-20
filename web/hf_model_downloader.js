import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXTENSION_NAME = "ComfyUI.HFModelDownloader";
const STYLE_ID = "hfmd-style";
const BUTTON_ID = "hfmd-sidebar-button";
const FLOATING_BUTTON_ID = "hfmd-floating-button";
const OVERLAY_ID = "hfmd-overlay";
const SIDEBAR_TAB_ID = "hf-model-downloader-sidebar-tab";
const LAUNCH_SHORTCUT = "Ctrl/Cmd+Shift+B";
const OWNER_SOURCES_STORAGE_KEY = "hfmd.owner_sources";
const UI_REFRESH_AGE_MS = 5 * 60 * 1000;
const ASSET_VERSION = "2.0.0";
const LAYOUT_VERSION = "v7";
const DENSITY_KEY = "hfmd.density";

function readStoredValue(key) {
    try {
        return localStorage.getItem(key) || "";
    } catch {
        return "";
    }
}

function writeStoredValue(key, value) {
    try {
        if (value) localStorage.setItem(key, value);
        else localStorage.removeItem(key);
    } catch {
        // ignore storage failures
    }
}

const state = {
    items: [],
    itemsById: new Map(),
    categories: [],
    categoryCounts: {},
    activeCategory: null,
    selectedIds: new Set(),
    search: "",
    familyFilter: "ALL",
    ownerFilter: "ALL",
    ownerSourcesInput: readStoredValue(OWNER_SOURCES_STORAGE_KEY),
    strictFilter: true,
    view: "browse",
    live: {
        query: "",
        sort: "trending",
        pipeline: "any",
        results: [],
        activeRepo: null,
        files: [],
        selected: new Set(),
        loading: false,
        debounce: null,
    },
    aria2: { installed: true, checked: false },
    settingsOpen: false,
    detailsItemId: null,
    indexMeta: {
        generatedAt: null,
        owners: [],
        defaultOwners: [],
        errors: [],
        installedCount: 0,
    },
    tokenMeta: {
        configured: false,
        source: "none",
        hint: "",
    },
    jobId: null,
    pollTimer: null,
    jobs: [],
    jobsPollTimer: null,
    ui: null,
    domObserver: null,
    sidebarTabRegistered: false,
    indexLoadedAtMs: 0,
    searchDebounce: null,
    compact: readStoredValue(DENSITY_KEY) === "compact",
};

function formatBytes(size) {
    if (!size || Number.isNaN(size)) return "?";
    const value = Number(size);
    if (value <= 0) return "?";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let n = value;
    let unit = 0;
    while (n >= 1024 && unit < units.length - 1) {
        n /= 1024;
        unit += 1;
    }
    return unit === 0 ? `${Math.round(n)} ${units[unit]}` : `${n.toFixed(1)} ${units[unit]}`;
}

function formatSpeed(bps) {
    const value = Number(bps || 0);
    if (!Number.isFinite(value) || value <= 0) return "";
    return `${formatBytes(value)}/s`;
}

function fmtDate(unixSeconds) {
    if (!unixSeconds) return "unknown";
    const d = new Date(Number(unixSeconds) * 1000);
    return d.toLocaleString();
}

function cssUrl() {
    return new URL("./hf_model_downloader.css", import.meta.url).href;
}

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const link = document.createElement("link");
    link.id = STYLE_ID;
    link.rel = "stylesheet";
    link.href = cssUrl();
    document.head.appendChild(link);
}

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
}

function option(value, label) {
    const node = document.createElement("option");
    node.value = value;
    node.textContent = label;
    return node;
}

function parseOwnerSources(raw) {
    const owners = [];
    const seen = new Set();
    for (const part of String(raw || "").split(",")) {
        const owner = part.trim();
        if (!owner) continue;
        const key = owner.toLowerCase();
        if (seen.has(key)) continue;
        seen.add(key);
        owners.push(owner);
    }
    return owners;
}

function prettyCategoryName(category) {
    return String(category || "")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (m) => m.toUpperCase());
}

function ensureOverlay() {
    const existing = document.getElementById(OVERLAY_ID);
    if (existing) {
        const hasCurrentLayout =
            existing.dataset.hfmdLayout === LAYOUT_VERSION &&
            existing.querySelector(".hfmd-live") &&
            existing.querySelector(".hfmd-downloads-panel") &&
            existing.querySelector(".hfmd-family-filter") &&
            existing.querySelector(".hfmd-owner-sources") &&
            existing.querySelector(".hfmd-apply-sources") &&
            existing.querySelector(".hfmd-settings-button") &&
            existing.querySelector(".hfmd-settings-panel") &&
            existing.querySelector(".hfmd-download-progress") &&
            existing.querySelector(".hfmd-strict-toggle");
        if (!hasCurrentLayout) {
            existing.remove();
        } else {
            state.ui = {
                overlay: existing,
                modal: existing.querySelector(".hfmd-modal"),
                closeButton: existing.querySelector(".hfmd-close"),
                settingsButton: existing.querySelector(".hfmd-settings-button"),
                browseButton: existing.querySelector(".hfmd-view-browse"),
                compactButton: existing.querySelector(".hfmd-density-toggle"),
                liveButton: existing.querySelector(".hfmd-view-live"),
                downloadsButton: existing.querySelector(".hfmd-view-downloads"),
                refreshButton: existing.querySelector(".hfmd-refresh"),
                toolbar: existing.querySelector(".hfmd-toolbar"),
                searchInput: existing.querySelector(".hfmd-search"),
                familyFilter: existing.querySelector(".hfmd-family-filter"),
                ownerFilter: existing.querySelector(".hfmd-owner-filter"),
                ownersInput: existing.querySelector(".hfmd-owner-sources"),
                ownersApplyButton: existing.querySelector(".hfmd-apply-sources"),
                strictToggle: existing.querySelector(".hfmd-strict-toggle"),
                settingsPanel: existing.querySelector(".hfmd-settings-panel"),
                settingsClose: existing.querySelector(".hfmd-settings-close"),
                tokenInput: existing.querySelector(".hfmd-token-input"),
                tokenSave: existing.querySelector(".hfmd-token-save"),
                tokenClear: existing.querySelector(".hfmd-token-clear"),
                tokenMeta: existing.querySelector(".hfmd-token-meta"),
                tabs: existing.querySelector(".hfmd-tabs"),
                body: existing.querySelector(".hfmd-body"),
                list: existing.querySelector(".hfmd-list"),
                details: existing.querySelector(".hfmd-details"),
                livePanel: existing.querySelector(".hfmd-live"),
                liveSearch: existing.querySelector(".hfmd-live-search"),
                liveSort: existing.querySelector(".hfmd-live-sort"),
                livePipeline: existing.querySelector(".hfmd-live-pipeline"),
                liveRepos: existing.querySelector(".hfmd-live-repos"),
                liveFiles: existing.querySelector(".hfmd-live-files"),
                liveAria2: existing.querySelector(".hfmd-aria2-warn"),
                liveAria2Text: existing.querySelector(".hfmd-aria2-warn span"),
                liveAria2Button: existing.querySelector(".hfmd-aria2-warn button"),
                downloadsPanel: existing.querySelector(".hfmd-downloads-panel"),
                downloadsList: existing.querySelector(".hfmd-downloads-list"),
                selectedInfo: existing.querySelector(".hfmd-selected-info"),
                status: existing.querySelector(".hfmd-status"),
                progressWrap: existing.querySelector(".hfmd-download-progress"),
                progressText: existing.querySelector(".hfmd-download-progress-text"),
                progressFill: existing.querySelector(".hfmd-download-progress-fill"),
                selectTabButton: existing.querySelector(".hfmd-select-tab"),
                clearTabButton: existing.querySelector(".hfmd-clear-tab"),
                clearAllButton: existing.querySelector(".hfmd-clear-all"),
                downloadButton: existing.querySelector(".hfmd-download"),
                maxConcurrentInput: existing.querySelector(".hfmd-max-concurrent"),
                connectionsInput: existing.querySelector(".hfmd-connections"),
                indexMeta: existing.querySelector(".hfmd-index-meta"),
            };
            updateOwnerSourcesUi();
            renderTokenMeta();
            applyCompact();
            return existing;
        }
    }

    const overlay = el("div", "hfmd-overlay");
    overlay.id = OVERLAY_ID;
    overlay.style.display = "none";

    const modal = el("div", "hfmd-modal");
    const header = el("div", "hfmd-header");
    const titleWrap = el("div", "hfmd-title-wrap");
    titleWrap.appendChild(el("h2", "hfmd-title", "HF Model Browser"));
    titleWrap.appendChild(
        el("p", "hfmd-subtitle", "Curated picks, or search the whole Hub live. Neon build."),
    );

    const headerActions = el("div", "hfmd-header-actions");
    const browseButton = el("button", "hfmd-btn hfmd-view-toggle hfmd-view-browse", "Browse");
    const liveButton = el("button", "hfmd-btn hfmd-view-toggle hfmd-view-live", "Live HF");
    const downloadsButton = el("button", "hfmd-btn hfmd-view-toggle hfmd-view-downloads", "Downloads");
    const compactButton = el("button", "hfmd-btn hfmd-density-toggle", "Compact");
    const refreshButton = el("button", "hfmd-btn hfmd-refresh", "Refresh Index");
    const settingsButton = el("button", "hfmd-btn hfmd-settings-button", "Settings");
    const closeButton = el("button", "hfmd-btn hfmd-close", "Close");
    headerActions.appendChild(browseButton);
    headerActions.appendChild(liveButton);
    headerActions.appendChild(downloadsButton);
    headerActions.appendChild(compactButton);
    headerActions.appendChild(refreshButton);
    headerActions.appendChild(settingsButton);
    headerActions.appendChild(closeButton);

    header.appendChild(titleWrap);
    header.appendChild(headerActions);

    const toolbar = el("div", "hfmd-toolbar");
    const searchInput = el("input", "hfmd-search");
    searchInput.type = "search";
    searchInput.placeholder = "Filter by model name, repo, family, path...";

    const familyFilter = el("select", "hfmd-filter-select hfmd-family-filter");
    familyFilter.appendChild(option("ALL", "All Types"));

    const ownerFilter = el("select", "hfmd-filter-select hfmd-owner-filter");
    ownerFilter.appendChild(option("ALL", "All Owners"));
    const ownersInput = el("input", "hfmd-owner-sources");
    ownersInput.type = "text";
    ownersInput.value = state.ownerSourcesInput || "";
    ownersInput.placeholder = "Sources (comma-separated owners)";
    const ownersApplyButton = el("button", "hfmd-btn hfmd-apply-sources", "Apply Sources");
    const strictToggle = el("button", "hfmd-btn hfmd-strict-toggle", "Strict: On");

    const selectTabButton = el("button", "hfmd-btn hfmd-select-tab", "Select Tab");
    const clearTabButton = el("button", "hfmd-btn hfmd-clear-tab", "Clear Tab");
    const clearAllButton = el("button", "hfmd-btn hfmd-clear-all", "Clear All");
    const selectedInfo = el("div", "hfmd-selected-info", "Selected: 0");

    toolbar.appendChild(searchInput);
    toolbar.appendChild(familyFilter);
    toolbar.appendChild(ownerFilter);
    toolbar.appendChild(ownersInput);
    toolbar.appendChild(ownersApplyButton);
    toolbar.appendChild(strictToggle);
    toolbar.appendChild(selectTabButton);
    toolbar.appendChild(clearTabButton);
    toolbar.appendChild(clearAllButton);
    toolbar.appendChild(selectedInfo);

    const tabs = el("div", "hfmd-tabs");

    const settingsPanel = el("div", "hfmd-settings-panel");
    settingsPanel.style.display = "none";
    settingsPanel.appendChild(el("h3", "hfmd-settings-title", "Hugging Face Settings"));
    settingsPanel.appendChild(
        el("p", "hfmd-settings-text", "Add your HF token for gated repositories (Black Forest Labs, etc)."),
    );
    const tokenInput = el("input", "hfmd-token-input");
    tokenInput.type = "password";
    tokenInput.placeholder = "hf_xxx...";
    tokenInput.autocomplete = "off";
    tokenInput.spellcheck = false;
    settingsPanel.appendChild(tokenInput);
    const tokenActions = el("div", "hfmd-settings-actions");
    const tokenSave = el("button", "hfmd-btn hfmd-token-save", "Save Token");
    const tokenClear = el("button", "hfmd-btn hfmd-token-clear", "Clear");
    const settingsClose = el("button", "hfmd-btn hfmd-settings-close", "Done");
    tokenActions.appendChild(tokenSave);
    tokenActions.appendChild(tokenClear);
    tokenActions.appendChild(settingsClose);
    settingsPanel.appendChild(tokenActions);
    const tokenMeta = el("div", "hfmd-token-meta", "Token: unknown");
    settingsPanel.appendChild(tokenMeta);

    const body = el("div", "hfmd-body");
    const list = el("div", "hfmd-list");
    const details = el("div", "hfmd-details");
    details.appendChild(el("div", "hfmd-details-empty", "Hover a model to see details."));
    body.appendChild(list);
    body.appendChild(details);

    const livePanel = el("div", "hfmd-live");
    livePanel.style.display = "none";
    const liveBar = el("div", "hfmd-live-bar");
    const liveSearch = el("input", "hfmd-live-search");
    liveSearch.placeholder = "Search all of Hugging Face — repo, author, keyword…";
    liveSearch.spellcheck = false;
    const liveSort = el("select", "hfmd-select hfmd-live-sort");
    [
        ["trending", "Trending"],
        ["downloads", "Most downloaded"],
        ["likes", "Most liked"],
        ["modified", "Recently updated"],
        ["created", "Newest"],
    ].forEach(([v, label]) => liveSort.appendChild(option(v, label)));
    const livePipeline = el("select", "hfmd-select hfmd-live-pipeline");
    [
        ["any", "Any task"],
        ["text-to-image", "Text to image"],
        ["image-to-image", "Image to image"],
        ["text-to-video", "Text to video"],
        ["image-to-video", "Image to video"],
    ].forEach(([v, label]) => livePipeline.appendChild(option(v, label)));
    const liveHint = el("div", "hfmd-live-hint", "Live from the Hub. No index, no cache wait.");
    liveBar.appendChild(liveSearch);
    liveBar.appendChild(liveSort);
    liveBar.appendChild(livePipeline);
    liveBar.appendChild(liveHint);

    const liveChips = el("div", "hfmd-live-chips");
    // The Hub's search ANDs its terms, so a two-word query like "upscale esrgan"
    // only matches repos containing both and returns near-empty junk. Keep every
    // quick search to a single token.
    const QUICK_SEARCHES = [
        ["Upscalers", "upscaler"],
        ["ESRGAN", "esrgan"],
        ["Krea 2", "krea"],
        ["ControlNet", "controlnet"],
        ["LoRAs", "lora"],
        ["VAE", "vae"],
        ["Encoders", "encoder"],
        ["GGUF", "gguf"],
    ];
    for (const [label, query] of QUICK_SEARCHES) {
        const chip = el("button", "hfmd-chip hfmd-chip-button", label);
        chip.type = "button";
        chip.addEventListener("click", () => {
            state.live.query = query;
            liveSearch.value = query;
            // Popularity is the only usable quality signal here: ESRGAN repos
            // report zero downloads, so sort by likes.
            state.live.sort = "likes";
            liveSort.value = "likes";
            runLiveSearch();
        });
        liveChips.appendChild(chip);
    }

    const liveAria2 = el("div", "hfmd-aria2-warn");
    liveAria2.style.display = "none";
    const liveAria2Text = el("span", "", "aria2c is missing — downloads cannot run without it.");
    const liveAria2Button = el("button", "", "Install aria2");
    liveAria2.appendChild(liveAria2Text);
    liveAria2.appendChild(liveAria2Button);

    const liveBody = el("div", "hfmd-live-body");
    const liveRepos = el("div", "hfmd-live-repos");
    const liveFiles = el("div", "hfmd-live-files");
    liveBody.appendChild(liveRepos);
    liveBody.appendChild(liveFiles);
    livePanel.appendChild(liveBar);
    livePanel.appendChild(liveChips);
    livePanel.appendChild(liveAria2);
    livePanel.appendChild(liveBody);

    const downloadsPanel = el("div", "hfmd-downloads-panel");
    downloadsPanel.style.display = "none";
    const downloadsList = el("div", "hfmd-downloads-list");
    downloadsPanel.appendChild(downloadsList);

    const footer = el("div", "hfmd-footer");
    const left = el("div", "hfmd-footer-left");
    const status = el("div", "hfmd-status", "Ready");
    const indexMeta = el("div", "hfmd-index-meta", "Cache: unknown");
    const progressWrap = el("div", "hfmd-download-progress");
    progressWrap.style.display = "none";
    const progressText = el("div", "hfmd-download-progress-text", "Preparing download...");
    const progressBar = el("div", "hfmd-download-progress-bar");
    const progressFill = el("div", "hfmd-download-progress-fill");
    progressBar.appendChild(progressFill);
    progressWrap.appendChild(progressText);
    progressWrap.appendChild(progressBar);
    left.appendChild(status);
    left.appendChild(indexMeta);
    left.appendChild(progressWrap);

    const right = el("div", "hfmd-footer-right");
    const concurrentWrap = el("label", "hfmd-number-wrap");
    concurrentWrap.appendChild(el("span", "", "Concurrent"));
    const maxConcurrentInput = el("input", "hfmd-max-concurrent");
    maxConcurrentInput.type = "number";
    maxConcurrentInput.min = "1";
    maxConcurrentInput.max = "32";
    maxConcurrentInput.value = "8";
    concurrentWrap.appendChild(maxConcurrentInput);

    const connectionsWrap = el("label", "hfmd-number-wrap");
    connectionsWrap.appendChild(el("span", "", "Connections"));
    const connectionsInput = el("input", "hfmd-connections");
    connectionsInput.type = "number";
    connectionsInput.min = "1";
    connectionsInput.max = "32";
    connectionsInput.value = "16";
    connectionsWrap.appendChild(connectionsInput);

    const downloadButton = el("button", "hfmd-btn hfmd-download", "Download Selected");
    right.appendChild(concurrentWrap);
    right.appendChild(connectionsWrap);
    right.appendChild(downloadButton);

    footer.appendChild(left);
    footer.appendChild(right);

    modal.appendChild(header);
    modal.appendChild(toolbar);
    modal.appendChild(settingsPanel);
    modal.appendChild(tabs);
    modal.appendChild(body);
    modal.appendChild(livePanel);
    modal.appendChild(downloadsPanel);
    modal.appendChild(footer);
    overlay.dataset.hfmdLayout = LAYOUT_VERSION;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    overlay.addEventListener("click", (event) => {
        if (event.target === overlay) closeModal();
    });
    closeButton.addEventListener("click", () => closeModal());
    settingsButton.addEventListener("click", async () => {
        const next = !state.settingsOpen;
        setSettingsOpen(next);
        if (next) {
            await fetchSettings();
        }
    });
    settingsClose.addEventListener("click", () => setSettingsOpen(false));
    tokenSave.addEventListener("click", () => saveTokenFromInput());
    tokenClear.addEventListener("click", () => clearToken());
    tokenInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        saveTokenFromInput();
    });
    compactButton.addEventListener("click", () => setCompact(!state.compact));
    browseButton.addEventListener("click", () => setView("browse"));
    liveButton.addEventListener("click", () => {
        setView("live");
        checkAria2();
        if (!state.live.results.length) runLiveSearch();
    });
    liveSearch.addEventListener("input", () => {
        state.live.query = liveSearch.value;
        clearTimeout(state.live.debounce);
        state.live.debounce = setTimeout(() => runLiveSearch(), 320);
    });
    liveSearch.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        clearTimeout(state.live.debounce);
        runLiveSearch();
    });
    liveSort.addEventListener("change", () => {
        state.live.sort = liveSort.value;
        runLiveSearch();
    });
    livePipeline.addEventListener("change", () => {
        state.live.pipeline = livePipeline.value;
        runLiveSearch();
    });
    liveAria2Button.addEventListener("click", () => installAria2());
    downloadsButton.addEventListener("click", () => {
        setView("downloads");
        fetchJobs();
    });
    refreshButton.addEventListener("click", () => fetchIndex(true));
    searchInput.addEventListener("input", () => {
        state.search = searchInput.value.toLowerCase().trim();
        clearTimeout(state.searchDebounce);
        state.searchDebounce = setTimeout(() => renderList(), 140);
    });
    familyFilter.addEventListener("change", () => {
        state.familyFilter = familyFilter.value || "ALL";
        renderList();
    });
    ownerFilter.addEventListener("change", () => {
        state.ownerFilter = ownerFilter.value || "ALL";
        renderList();
    });
    const applyOwnerSources = () => {
        state.ownerSourcesInput = (ownersInput.value || "").trim();
        writeStoredValue(OWNER_SOURCES_STORAGE_KEY, state.ownerSourcesInput);
        const owners = parseOwnerSources(state.ownerSourcesInput);
        const sourceLabel = owners.length ? owners.join(", ") : "default owners";
        setStatus(`Using sources: ${sourceLabel}`, "info");
        fetchIndex(true);
    };
    ownersApplyButton.addEventListener("click", () => applyOwnerSources());
    ownersInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        applyOwnerSources();
    });
    strictToggle.addEventListener("click", () => {
        state.strictFilter = !state.strictFilter;
        updateStrictToggleLabel();
        fetchIndex(true);
    });
    selectTabButton.addEventListener("click", () => {
        let added = 0;
        for (const item of currentVisibleItems()) {
            if (item.installed) continue;
            if (!state.selectedIds.has(item.id)) added += 1;
            state.selectedIds.add(item.id);
        }
        updateSelectionSummary();
        renderList();
        setStatus(`Selected ${added} downloadable models in current tab.`, "success");
    });
    clearTabButton.addEventListener("click", () => {
        for (const item of currentVisibleItems()) state.selectedIds.delete(item.id);
        updateSelectionSummary();
        renderList();
        setStatus("Cleared selection in current tab.", "info");
    });
    clearAllButton.addEventListener("click", () => {
        state.selectedIds.clear();
        updateSelectionSummary();
        renderList();
        setStatus("Selection cleared.", "info");
    });
    downloadButton.addEventListener("click", () => {
        if (state.view === "live") startLiveDownload();
        else startDownload();
    });
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && overlay.style.display !== "none") closeModal();
        if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "b") {
            event.preventDefault();
            openModal();
        }
    });

    state.ui = {
        overlay,
        modal,
        closeButton,
        settingsButton,
        browseButton,
        compactButton,
        liveButton,
        downloadsButton,
        refreshButton,
        toolbar,
        searchInput,
        familyFilter,
        ownerFilter,
        ownersInput,
        ownersApplyButton,
        strictToggle,
        settingsPanel,
        settingsClose,
        tokenInput,
        tokenSave,
        tokenClear,
        tokenMeta,
        tabs,
        body,
        list,
        details,
        livePanel,
        liveSearch,
        liveSort,
        livePipeline,
        liveRepos,
        liveFiles,
        liveAria2,
        liveAria2Text,
        liveAria2Button,
        downloadsPanel,
        downloadsList,
        selectedInfo,
        status,
        progressWrap,
        progressText,
        progressFill,
        selectTabButton,
        clearTabButton,
        clearAllButton,
        downloadButton,
        maxConcurrentInput,
        connectionsInput,
        indexMeta,
    };
    setView("browse");
    updateStrictToggleLabel();
    updateOwnerSourcesUi();
    renderTokenMeta();
    return overlay;
}

function setStatus(text, kind = "info") {
    if (!state.ui?.status) return;
    state.ui.status.textContent = text;
    state.ui.status.className = `hfmd-status hfmd-status-${kind}`;
}

function updateStrictToggleLabel() {
    if (!state.ui?.strictToggle) return;
    state.ui.strictToggle.textContent = `Strict: ${state.strictFilter ? "On" : "Off"}`;
    state.ui.strictToggle.classList.toggle("is-off", !state.strictFilter);
}

function updateOwnerSourcesUi() {
    if (!state.ui?.ownersInput) return;
    const defaults = Array.isArray(state.indexMeta.defaultOwners) ? state.indexMeta.defaultOwners : [];
    const fallback = defaults.length ? defaults.join(", ") : "Comfy-Org, Kijai, black-forest-labs";
    state.ui.ownersInput.placeholder = `Sources: ${fallback}`;
    if (document.activeElement !== state.ui.ownersInput) {
        state.ui.ownersInput.value = state.ownerSourcesInput || "";
    }
}

function renderTokenMeta() {
    if (!state.ui?.tokenMeta) return;
    const configured = Boolean(state.tokenMeta.configured);
    if (!configured) {
        state.ui.tokenMeta.textContent = "Token: not configured";
        return;
    }
    const source = state.tokenMeta.source || "unknown";
    const hint = state.tokenMeta.hint || "";
    state.ui.tokenMeta.textContent = `Token: ${hint} (${source})`;
}

function setSettingsOpen(open) {
    state.settingsOpen = Boolean(open);
    if (!state.ui?.settingsPanel || !state.ui?.settingsButton) return;
    state.ui.settingsPanel.style.display = state.settingsOpen ? "grid" : "none";
    state.ui.settingsButton.classList.toggle("is-active", state.settingsOpen);
}

async function fetchSettings() {
    try {
        const response = await api.fetchApi("/hf-model-downloader/settings");
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Settings request failed (${response.status})`);
        }
        state.tokenMeta = {
            configured: Boolean(payload.token_configured),
            source: payload.token_source || "none",
            hint: payload.token_hint || "",
        };
        renderTokenMeta();
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed loading settings`, error);
        setStatus(`Settings load failed: ${error.message || error}`, "error");
    }
}

async function saveTokenFromInput() {
    if (!state.ui?.tokenInput) return;
    const token = (state.ui.tokenInput.value || "").trim();
    if (!token) {
        setStatus("Token input is empty.", "warn");
        return;
    }
    try {
        const response = await api.fetchApi("/hf-model-downloader/token", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token }),
        });
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Save token failed (${response.status})`);
        }
        state.ui.tokenInput.value = "";
        state.tokenMeta = {
            configured: Boolean(payload.token_configured),
            source: payload.token_source || "file",
            hint: payload.token_hint || "",
        };
        renderTokenMeta();
        setStatus(payload.message || "HF token saved.", "success");
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed saving HF token`, error);
        setStatus(`Token save failed: ${error.message || error}`, "error");
    }
}

async function clearToken() {
    try {
        const response = await api.fetchApi("/hf-model-downloader/token", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: "" }),
        });
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Clear token failed (${response.status})`);
        }
        if (state.ui?.tokenInput) state.ui.tokenInput.value = "";
        state.tokenMeta = { configured: false, source: "none", hint: "" };
        renderTokenMeta();
        setStatus(payload.message || "HF token cleared.", "warn");
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed clearing HF token`, error);
        setStatus(`Token clear failed: ${error.message || error}`, "error");
    }
}

function computeProgressPercent(payload) {
    const explicit = Number(payload?.progress);
    if (Number.isFinite(explicit) && explicit >= 0) {
        return Math.max(0, Math.min(100, explicit));
    }
    const downloaded = Number(payload?.downloaded_bytes || 0);
    const totalBytes = Number(payload?.total_bytes || 0);
    if (totalBytes > 0) {
        return Math.max(0, Math.min(100, (downloaded / totalBytes) * 100));
    }
    const completed = Number(payload?.completed || 0);
    const total = Number(payload?.total || 0);
    if (total > 0) {
        return Math.max(0, Math.min(100, (completed / total) * 100));
    }
    return 0;
}

function renderBottomProgress(payload) {
    if (!state.ui?.progressWrap || !state.ui?.progressText || !state.ui?.progressFill) return;
    const status = String(payload?.status || "");
    if (!payload || status === "done" || status === "error") {
        state.ui.progressWrap.style.display = "none";
        state.ui.progressFill.style.width = "0%";
        return;
    }
    const percent = computeProgressPercent(payload);
    const total = Number(payload?.total || 0);
    const completed = Number(payload?.completed || 0);
    const downloaded = Number(payload?.downloaded_bytes || 0);
    const totalBytes = Number(payload?.total_bytes || 0);
    const bytesPart = totalBytes > 0 ? `${formatBytes(downloaded)} / ${formatBytes(totalBytes)}` : `${completed}/${total} files`;
    const label = status === "queued" ? "Queued…" : "Downloading…";
    state.ui.progressText.textContent = `${label} ${bytesPart} · ${Math.round(percent)}%`;
    state.ui.progressFill.style.width = `${percent}%`;
    state.ui.progressWrap.style.display = "grid";
}

function setCompact(on) {
    state.compact = Boolean(on);
    writeStoredValue(DENSITY_KEY, state.compact ? "compact" : "");
    applyCompact();
}

function applyCompact() {
    const ui = state.ui;
    if (!ui?.modal) return;
    ui.modal.classList.toggle("hfmd-compact", state.compact);
    if (ui.compactButton) {
        ui.compactButton.classList.toggle("is-active", state.compact);
        ui.compactButton.textContent = state.compact ? "Comfortable" : "Compact";
    }
}

function setView(view) {
    const known = ["browse", "live", "downloads"];
    state.view = known.includes(view) ? view : "browse";
    if (!state.ui) return;
    const browse = state.view === "browse";
    const live = state.view === "live";
    const downloads = state.view === "downloads";
    state.ui.toolbar.style.display = browse ? "flex" : "none";
    state.ui.tabs.style.display = browse ? "flex" : "none";
    state.ui.body.style.display = browse ? "grid" : "none";
    state.ui.livePanel.style.display = live ? "flex" : "none";
    // flex, not block: the list needs a flex parent to scroll inside.
    state.ui.downloadsPanel.style.display = downloads ? "flex" : "none";
    state.ui.browseButton.classList.toggle("is-active", browse);
    state.ui.liveButton.classList.toggle("is-active", live);
    state.ui.downloadsButton.classList.toggle("is-active", downloads);
    updateSelectionSummary();
    if (downloads) {
        startJobsPolling();
    } else if (!state.jobId) {
        stopJobsPolling();
    }
}

function groupedByFamily(items) {
    const groups = new Map();
    for (const item of items) {
        const family = item.family || "MISC";
        if (!groups.has(family)) groups.set(family, []);
        groups.get(family).push(item);
    }
    return new Map([...groups.entries()].sort((a, b) => a[0].localeCompare(b[0])));
}

function categoryCount(category) {
    return state.categoryCounts?.[category] || 0;
}

function populateFilterSelect(selectEl, values, currentValue, allLabel) {
    selectEl.innerHTML = "";
    selectEl.appendChild(option("ALL", allLabel));
    for (const value of values) {
        selectEl.appendChild(option(value, value));
    }
    const selected = values.includes(currentValue) ? currentValue : "ALL";
    selectEl.value = selected;
    return selected;
}

function renderFilterControls() {
    if (!state.ui?.familyFilter || !state.ui?.ownerFilter) return;
    const itemsInCategory = state.items.filter((item) => item.category === state.activeCategory);
    const families = [...new Set(itemsInCategory.map((item) => item.family || "MISC"))].sort((a, b) =>
        a.localeCompare(b),
    );
    const owners = [...new Set(itemsInCategory.map((item) => item.owner || "?"))].sort((a, b) => a.localeCompare(b));
    state.familyFilter = populateFilterSelect(state.ui.familyFilter, families, state.familyFilter, "All Types");
    state.ownerFilter = populateFilterSelect(state.ui.ownerFilter, owners, state.ownerFilter, "All Owners");
}

function renderTabs() {
    const tabsEl = state.ui.tabs;
    tabsEl.innerHTML = "";
    if (!state.categories.length) {
        tabsEl.appendChild(el("div", "hfmd-tabs-empty", "No categories found."));
        return;
    }
    for (const category of state.categories) {
        const button = el("button", "hfmd-tab");
        if (category === state.activeCategory) button.classList.add("is-active");
        button.textContent = `${prettyCategoryName(category)} (${categoryCount(category)})`;
        button.addEventListener("click", () => {
            state.activeCategory = category;
            state.detailsItemId = null;
            renderTabs();
            renderFilterControls();
            renderList();
        });
        tabsEl.appendChild(button);
    }
}

function currentVisibleItems() {
    if (!state.activeCategory) return [];
    const term = state.search;
    return state.items.filter((item) => {
        if (item.category !== state.activeCategory) return false;
        if (state.familyFilter !== "ALL" && item.family !== state.familyFilter) return false;
        if (state.ownerFilter !== "ALL" && item.owner !== state.ownerFilter) return false;
        if (!term) return true;
        const hay = `${item.title} ${item.repo_id} ${item.path} ${item.family} ${item.category}`.toLowerCase();
        return hay.includes(term);
    });
}

function renderDetail(item) {
    state.detailsItemId = item?.id || null;
    const details = state.ui.details;
    details.innerHTML = "";
    if (!item) {
        details.appendChild(el("div", "hfmd-details-empty", "Hover a model to see details."));
        return;
    }
    const card = el("div", "hfmd-details-card");
    const title = el("h3", "hfmd-details-title", item.title || item.filename);
    const rows = [
        ["Type", item.family || "MISC"],
        ["Category", prettyCategoryName(item.category || "?")],
        ["Repo", item.repo_id || "?"],
        ["Path", item.path || "?"],
        ["Size", formatBytes(item.size)],
        ["Downloads", String(item.downloads || 0)],
        ["Likes", String(item.likes || 0)],
        ["Installed", item.installed ? "Yes" : "No"],
        ["Installed Path", item.installed_path || "-"],
        ["Target", item.target_preview || "?"],
    ];
    card.appendChild(title);
    for (const [label, value] of rows) {
        const row = el("div", "hfmd-details-row");
        row.appendChild(el("span", "hfmd-details-label", label));
        row.appendChild(el("code", "hfmd-details-value", value));
        card.appendChild(row);
    }
    details.appendChild(card);
}

function renderList() {
    const listEl = state.ui.list;
    listEl.innerHTML = "";
    const visible = currentVisibleItems();

    if (!visible.length) {
        listEl.appendChild(el("div", "hfmd-list-empty", "No models match this filter set."));
        renderDetail(null);
        updateSelectionSummary();
        return;
    }

    const grouped = groupedByFamily(visible);
    for (const [family, items] of grouped) {
        const section = el("div", "hfmd-family");
        const header = el("div", "hfmd-family-header", `${family} (${items.length})`);
        section.appendChild(header);

        for (const item of items) {
            const row = el("div", "hfmd-row");
            row.dataset.itemId = item.id;
            const isInstalled = Boolean(item.installed);
            if (isInstalled) {
                row.classList.add("is-installed");
                state.selectedIds.delete(item.id);
            }
            if (state.selectedIds.has(item.id)) row.classList.add("is-selected");
            if (state.detailsItemId === item.id) row.classList.add("is-focused");

            const checkbox = el("input", "hfmd-row-check");
            checkbox.type = "checkbox";
            checkbox.checked = state.selectedIds.has(item.id);
            checkbox.disabled = isInstalled;
            checkbox.addEventListener("change", () => {
                if (isInstalled) {
                    checkbox.checked = false;
                    return;
                }
                if (checkbox.checked) state.selectedIds.add(item.id);
                else state.selectedIds.delete(item.id);
                row.classList.toggle("is-selected", checkbox.checked);
                updateSelectionSummary();
            });

            const meta = el("div", "hfmd-row-meta");
            const titleLine = el("div", "hfmd-row-title");
            titleLine.textContent = item.title || item.filename;
            if (isInstalled) {
                titleLine.appendChild(el("span", "hfmd-installed-pill", "Installed"));
            }
            meta.appendChild(titleLine);

            const subBits = [item.owner, formatBytes(item.size), `d:${item.downloads || 0}`];
            if (isInstalled) subBits.push("already in models dir");
            meta.appendChild(el("div", "hfmd-row-sub", subBits.join(" · ")));

            row.addEventListener("mouseenter", () => {
                renderDetail(item);
                for (const other of listEl.querySelectorAll(".hfmd-row.is-focused")) other.classList.remove("is-focused");
                row.classList.add("is-focused");
            });
            row.addEventListener("click", (event) => {
                if (isInstalled) return;
                if (event.target === checkbox) return;
                checkbox.checked = !checkbox.checked;
                checkbox.dispatchEvent(new Event("change"));
            });

            row.appendChild(checkbox);
            row.appendChild(meta);
            section.appendChild(row);
        }
        listEl.appendChild(section);
    }
    updateSelectionSummary();
}

function updateSelectionSummary() {
    if (!state.ui?.selectedInfo) return;
    if (state.view === "live") {
        const repo = state.live.activeRepo || "no repo";
        state.ui.selectedInfo.textContent =
            `Live: ${state.live.selected.size} selected of ${state.live.files.length} files · ${repo}`;
        return;
    }
    const total = state.items.length;
    const selected = state.selectedIds.size;
    const visibleItems = currentVisibleItems();
    const visible = visibleItems.length;
    const installedVisible = visibleItems.filter((item) => item.installed).length;
    state.ui.selectedInfo.textContent = `Selected: ${selected} / ${total} · Visible: ${visible} · Installed: ${installedVisible}`;
}

function rebuildItemsById() {
    state.itemsById = new Map();
    for (const item of state.items) {
        state.itemsById.set(item.id, item);
    }
}

async function fetchIndex(refresh = false) {
    setStatus(refresh ? "Refreshing curated index..." : "Loading curated index...", "info");
    try {
        const selectedOwners = parseOwnerSources(state.ownerSourcesInput);
        const strict = state.strictFilter ? 1 : 0;
        const query = new URLSearchParams({
            refresh: refresh ? "1" : "0",
            strict_filter: String(strict),
        });
        if (selectedOwners.length) {
            query.set("owners", selectedOwners.join(","));
        }
        const response = await api.fetchApi(`/hf-model-downloader/index?${query.toString()}`);
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Index request failed (${response.status})`);
        }

        state.items = Array.isArray(payload.items) ? payload.items : [];
        state.categories = Array.isArray(payload.categories) ? payload.categories : [];
        state.categoryCounts = payload.category_counts || {};
        state.indexMeta = {
            generatedAt: payload.generated_at || null,
            owners: payload.owners || [],
            defaultOwners: payload.default_owners || [],
            errors: payload.errors || [],
            installedCount: Number(payload.installed_count || 0),
        };
        state.indexLoadedAtMs = Date.now();
        state.strictFilter = payload.strict_filter !== false;
        updateStrictToggleLabel();
        updateOwnerSourcesUi();
        rebuildItemsById();

        if (!state.categories.includes(state.activeCategory)) {
            state.activeCategory = state.categories[0] || null;
        }

        if (state.selectedIds.size) {
            for (const id of [...state.selectedIds]) {
                const item = state.itemsById.get(id);
                if (!item || item.installed) state.selectedIds.delete(id);
            }
        }

        const ownersLabel = state.indexMeta.owners?.length ? state.indexMeta.owners.join(", ") : "n/a";
        state.ui.indexMeta.textContent = `Cache: ${fmtDate(state.indexMeta.generatedAt)} · Installed: ${state.indexMeta.installedCount} · Strict: ${state.strictFilter ? "On" : "Off"} · Owners: ${ownersLabel}`;
        renderTabs();
        renderFilterControls();
        renderList();
        setStatus(`Indexed ${state.items.length} curated model files.`, "success");
        if (state.indexMeta.errors?.length) {
            setStatus(`Indexed with ${state.indexMeta.errors.length} warning(s).`, "warn");
        }
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed to load index`, error);
        setStatus(`Index load failed: ${error.message || error}`, "error");
    }
}

function renderJobs() {
    if (!state.ui?.downloadsList) return;
    const root = state.ui.downloadsList;
    root.innerHTML = "";
    if (!state.jobs.length) {
        root.appendChild(el("div", "hfmd-list-empty", "No download jobs yet."));
        return;
    }

    for (const job of state.jobs) {
        const card = el("div", "hfmd-job-card");
        const status = String(job.status || "unknown");
        card.classList.add(`is-${status}`);

        const header = el("div", "hfmd-job-header");
        header.appendChild(el("div", "hfmd-job-id", `Job ${job.job_id || "?"}`));
        const right = el("div", "hfmd-job-header-right");
        right.appendChild(el("div", `hfmd-job-status hfmd-job-status-${status}`, status.toUpperCase()));
        if (status === "queued" || status === "running") {
            const cancelBtn = el("button", "hfmd-btn hfmd-job-cancel", "Cancel");
            cancelBtn.addEventListener("click", (event) => {
                event.stopPropagation();
                cancelJob(job.job_id);
            });
            right.appendChild(cancelBtn);
        }
        header.appendChild(right);
        card.appendChild(header);

        const total = Number(job.total || 0);
        const completed = Number(job.completed || 0);
        const downloadedBytes = Number(job.downloaded_bytes || 0);
        const totalBytes = Number(job.total_bytes || 0);
        const percent =
            totalBytes > 0
                ? Math.max(0, Math.min(100, (downloadedBytes / totalBytes) * 100))
                : total > 0
                  ? Math.max(0, Math.min(100, (completed / total) * 100))
                  : status === "done"
                    ? 100
                    : 0;
        const jobBytes = totalBytes > 0 ? `${formatBytes(downloadedBytes)} / ${formatBytes(totalBytes)}` : `${completed}/${total} files`;
        card.appendChild(el("div", "hfmd-job-progress-text", `${jobBytes} · ${Math.round(percent)}%`));

        const bar = el("div", "hfmd-job-progress");
        const fill = el("div", "hfmd-job-progress-fill");
        fill.style.width = `${percent}%`;
        bar.appendChild(fill);
        card.appendChild(bar);

        const files = Array.isArray(job.files) ? [...job.files] : [];
        if (files.length) {
            files.sort((a, b) => Number(a.index || 0) - Number(b.index || 0));
            const filesWrap = el("div", "hfmd-job-files");
            for (const file of files) {
                const fileStatus = String(file.status || "queued");
                const row = el("div", "hfmd-job-file");
                row.classList.add(`is-${fileStatus}`);

                const title = el("div", "hfmd-job-file-title", file.title || file.filename || "model");
                const statusPill = el("span", `hfmd-job-file-status hfmd-job-file-status-${fileStatus}`, fileStatus.toUpperCase());
                title.appendChild(statusPill);
                row.appendChild(title);

                const fileProgress = Number(file.progress || 0);
                const fileBar = el("div", "hfmd-job-file-bar");
                const fileFill = el("div", "hfmd-job-file-fill");
                fileFill.style.width = `${Math.max(0, Math.min(100, fileProgress))}%`;
                fileBar.appendChild(fileFill);
                row.appendChild(fileBar);

                const fileDownloaded = Number(file.downloaded_bytes || 0);
                const fileTotal = Number(file.total_bytes || 0);
                const speed = formatSpeed(file.speed_bps || 0);
                const bytesText =
                    fileTotal > 0
                        ? `${formatBytes(fileDownloaded)} / ${formatBytes(fileTotal)}`
                        : `${formatBytes(fileDownloaded)}`;
                const metaText = speed ? `${bytesText} · ${Math.round(fileProgress)}% · ${speed}` : `${bytesText} · ${Math.round(fileProgress)}%`;
                row.appendChild(el("div", "hfmd-job-file-meta", metaText));
                if (file.error) {
                    row.appendChild(el("div", "hfmd-job-file-error", String(file.error)));
                }
                filesWrap.appendChild(row);
            }
            card.appendChild(filesWrap);
        }

        const message = String(job.error || job.message || "");
        if (message) card.appendChild(el("div", "hfmd-job-message", message));

        const targets = Array.isArray(job.targets) ? job.targets : [];
        if (targets.length) {
            card.appendChild(el("div", "hfmd-job-target", targets[0]));
        }

        root.appendChild(card);
    }
}

async function cancelJob(jobId) {
    if (!jobId) return;
    try {
        const response = await api.fetchApi("/hf-model-downloader/cancel", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ job_id: jobId }),
        });
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Cancel failed (${response.status})`);
        }
        if (state.jobId === jobId) {
            setStatus(`Job ${jobId}: cancelling...`, "warn");
        }
        fetchJobs();
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed cancelling job`, error);
        setStatus(`Cancel failed: ${error.message || error}`, "error");
    }
}

async function fetchJobs() {
    try {
        const response = await api.fetchApi("/hf-model-downloader/jobs?limit=40");
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Jobs request failed (${response.status})`);
        }
        state.jobs = Array.isArray(payload.jobs) ? payload.jobs : [];
        renderJobs();
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed fetching jobs`, error);
        setStatus(`Jobs refresh failed: ${error.message || error}`, "error");
    }
}

async function pollJobStatus() {
    if (!state.jobId) return;
    try {
        const response = await api.fetchApi(`/hf-model-downloader/status?job_id=${encodeURIComponent(state.jobId)}`);
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Status request failed (${response.status})`);
        }
        const status = payload.status;
        const total = payload.total || 0;
        const completed = payload.completed || 0;
        if (status === "queued") {
            setStatus(`Job ${state.jobId}: queued (${total} files).`, "info");
            renderBottomProgress(payload);
        } else if (status === "running") {
            setStatus(`Job ${state.jobId}: running (${completed}/${total}).`, "info");
            renderBottomProgress(payload);
        } else if (status === "cancelled") {
            setStatus(payload.message || `Job ${state.jobId}: cancelled.`, "warn");
            renderBottomProgress(payload);
            stopPolling();
            setTimeout(() => renderBottomProgress(null), 900);
            fetchIndex(false);
        } else if (status === "done") {
            setStatus(payload.message || `Job ${state.jobId}: complete.`, "success");
            renderBottomProgress(payload);
            stopPolling();
            setTimeout(() => renderBottomProgress(null), 900);
            fetchIndex(false);
        } else if (status === "error") {
            setStatus(payload.error || payload.message || `Job ${state.jobId}: failed.`, "error");
            renderBottomProgress(payload);
            stopPolling();
            setTimeout(() => renderBottomProgress(null), 1200);
        } else {
            setStatus(`Job ${state.jobId}: ${status}`, "info");
            renderBottomProgress(payload);
        }
        if (state.view === "downloads") fetchJobs();
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed polling job status`, error);
        setStatus(`Status poll failed: ${error.message || error}`, "error");
        stopPolling();
        renderBottomProgress(null);
    }
}

function stopPolling() {
    if (state.pollTimer) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
    }
}

function startPolling() {
    stopPolling();
    state.pollTimer = setInterval(pollJobStatus, 1200);
    pollJobStatus();
}

function stopJobsPolling() {
    if (state.jobsPollTimer) {
        clearInterval(state.jobsPollTimer);
        state.jobsPollTimer = null;
    }
}

function hasLiveJobs() {
    return (state.jobs || []).some((job) =>
        ["queued", "running", "downloading", "starting"].includes(String(job.status || "")),
    );
}

function startJobsPolling() {
    if (state.jobsPollTimer) return;
    state.jobsPollTimer = setInterval(() => {
        // Idle polling every 1.5s kept the UI busy for no reason. Once nothing is
        // running, back off to an occasional refresh.
        if (hasLiveJobs() || state.jobId) {
            fetchJobs();
            state.jobsIdleTicks = 0;
            return;
        }
        state.jobsIdleTicks = (state.jobsIdleTicks || 0) + 1;
        if (state.jobsIdleTicks % 8 === 0) fetchJobs();
    }, 1500);
    fetchJobs();
}

async function startDownload() {
    const requestedIds = [...state.selectedIds];
    if (!requestedIds.length) {
        setStatus("No models selected.", "warn");
        return;
    }
    const ids = requestedIds.filter((id) => !state.itemsById.get(id)?.installed);
    const skippedInstalled = requestedIds.length - ids.length;
    if (!ids.length) {
        setStatus("All selected models are already installed.", "warn");
        return;
    }

    const maxConcurrent = Math.max(1, Math.min(32, Number(state.ui.maxConcurrentInput.value) || 8));
    const connections = Math.max(1, Math.min(32, Number(state.ui.connectionsInput.value) || 16));
    const selectedOwners = parseOwnerSources(state.ownerSourcesInput);
    setStatus(`Submitting ${ids.length} model(s) for download...`, "info");
    try {
        const body = {
            ids,
            max_concurrent_downloads: maxConcurrent,
            connections_per_download: connections,
            strict_filter: state.strictFilter,
        };
        if (selectedOwners.length) {
            body.owners = selectedOwners;
        }
        const response = await api.fetchApi("/hf-model-downloader/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const payload = await response.json();
        if (!response.ok || !payload?.ok) {
            throw new Error(payload?.error || `Download request failed (${response.status})`);
        }
        state.jobId = payload.job_id;
        const skipMsg = skippedInstalled > 0 ? ` (${skippedInstalled} already-installed skipped)` : "";
        setStatus(`Job ${state.jobId} started for ${payload.total} item(s).${skipMsg}`, "success");
        renderBottomProgress({
            status: "queued",
            total: payload.total,
            completed: 0,
            downloaded_bytes: 0,
            total_bytes: 0,
            progress: 0,
        });
        if (state.view === "downloads") startJobsPolling();
        startPolling();
    } catch (error) {
        console.error(`[${EXTENSION_NAME}] Failed starting download`, error);
        setStatus(`Download start failed: ${error.message || error}`, "error");
        renderBottomProgress(null);
    }
}

function openModal() {
    ensureStyles();
    ensureOverlay();
    state.ui.overlay.style.display = "flex";
    // Covers a freshly built overlay as well as a rehydrated one.
    applyCompact();
    setView(state.view);
    if (!state.items.length) {
        fetchIndex(false);
    } else {
        renderTabs();
        renderFilterControls();
        renderList();
        if (Date.now() - Number(state.indexLoadedAtMs || 0) > UI_REFRESH_AGE_MS) {
            fetchIndex(false);
        }
    }
    if (state.view === "downloads") {
        startJobsPolling();
    }
}

function closeModal() {
    if (!state.ui?.overlay) return;
    state.ui.overlay.style.display = "none";
    setSettingsOpen(false);
    stopJobsPolling();
    stopPolling();
    renderBottomProgress(null);
}

function findSidebarHost() {
    return (
        document.querySelector(".comfyui-button-group") ||
        document.querySelector(".comfy-menu .comfyui-button-group") ||
        document.querySelector(".comfy-menu") ||
        document.querySelector(".comfyui-menu-right") ||
        document.querySelector(".comfyui-menu")
    );
}

function createLaunchButton(className, compact = false) {
    const button = el("button", className);
    button.type = "button";
    button.setAttribute("aria-label", "Open Hugging Face Model Downloader");
    button.title = "Open Hugging Face Model Downloader";
    button.innerHTML = compact
        ? `<span class="hfmd-launch-icon">⬇</span><span class="hfmd-launch-text">Model Browser</span>`
        : `<span class="hfmd-launch-icon">⬇</span><span class="hfmd-launch-text">HF Models</span>`;
    button.addEventListener("click", () => openModal());
    return button;
}

function applyFloatingFallbackStyles(button) {
    button.style.position = "";
    button.style.right = "";
    button.style.bottom = "";
    button.style.zIndex = "";
    button.style.border = "";
    button.style.background = "";
    button.style.color = "";
    button.style.borderRadius = "";
    button.style.padding = "";
    button.style.fontWeight = "";
    button.style.display = "";
    button.style.alignItems = "";
    button.style.gap = "";
    button.style.cursor = "";
}

function ensureFloatingButton() {
    let floating = document.getElementById(FLOATING_BUTTON_ID);
    if (!floating) {
        floating = createLaunchButton("hfmd-floating-button", true);
        floating.id = FLOATING_BUTTON_ID;
        document.body.appendChild(floating);
    }
    applyFloatingFallbackStyles(floating);
    return floating;
}

function ensureSidebarButton(retry = 0) {
    let button = document.getElementById(BUTTON_ID);
    if (button) return button;
    const host = findSidebarHost();
    if (!host) {
        ensureFloatingButton();
        if (retry < 50) setTimeout(() => ensureSidebarButton(retry + 1), 300);
        return null;
    }

    button = createLaunchButton("hfmd-sidebar-button");
    button.id = BUTTON_ID;
    host.appendChild(button);
    return button;
}

function ensureSidebarTab() {
    if (state.sidebarTabRegistered) return;
    const manager = app.extensionManager;
    if (!manager || typeof manager.registerSidebarTab !== "function") return;
    manager.registerSidebarTab({
        id: SIDEBAR_TAB_ID,
        title: "HF Models",
        icon: "pi pi-download",
        type: "custom",
        render: (container) => {
            container.innerHTML = "";
            const wrap = el("div", "hfmd-sidebar-tab");
            wrap.appendChild(el("h3", "hfmd-sidebar-tab-title", "HF Model Browser"));
            wrap.appendChild(
                el("p", "hfmd-sidebar-tab-text", "Curated Hugging Face browser with launcher and live install tracking."),
            );
            const launch = createLaunchButton("hfmd-sidebar-tab-button", false);
            launch.querySelector(".hfmd-launch-text").textContent = "Open Model Browser";
            wrap.appendChild(launch);
            wrap.appendChild(el("p", "hfmd-sidebar-tab-hint", `Shortcut: ${LAUNCH_SHORTCUT}`));
            container.appendChild(wrap);
        },
    });
    state.sidebarTabRegistered = true;
}

function ensureDomObserver() {
    if (state.domObserver) return;
    // Observing document.body with subtree:true fires on every canvas repaint and
    // visibly lags the graph. The launch buttons only ever need re-adding when the
    // sidebar itself is rebuilt, so watch that host and fall back to a slow poll.
    const host = findSidebarHost();
    if (host) {
        state.domObserver = new MutationObserver(() => {
            ensureSidebarButton();
            ensureFloatingButton();
        });
        state.domObserver.observe(host, { childList: true });
        return;
    }
    state.domObserver = {
        timer: setInterval(() => {
            ensureSidebarButton();
            ensureFloatingButton();
        }, 4000),
        disconnect() {
            clearInterval(this.timer);
        },
    };
}

app.registerExtension({
    name: EXTENSION_NAME,
    actionBarButtons: [
        {
            icon: "icon-[mdi--download-box] size-4",
            tooltip: "HF Model Downloader",
            onClick: () => openModal(),
        },
    ],
    async setup() {
        ensureStyles();
        ensureOverlay();
        ensureFloatingButton();
        ensureSidebarButton();
        ensureSidebarTab();
        ensureDomObserver();
        checkAria2();
        window.openHFModelDownloader = openModal;
        console.log(`[${EXTENSION_NAME}] popup ready. Shortcut: ${LAUNCH_SHORTCUT}`);
    },
});

/* ============================================================
   Live Hugging Face browser
   ============================================================ */

function liveSelectedFiles() {
    return state.live.files.filter((file) => state.live.selected.has(file.id));
}

async function checkAria2() {
    try {
        const res = await api.fetchApi("/hf-model-downloader/aria2");
        const data = await res.json();
        state.aria2 = { installed: Boolean(data.installed), checked: true, version: data.version || "" };
    } catch (_) {
        state.aria2 = { installed: true, checked: false };
    }
    renderAria2Warning();
}

function renderAria2Warning() {
    const ui = state.ui;
    if (!ui?.liveAria2) return;
    const missing = state.aria2.checked && !state.aria2.installed;
    ui.liveAria2.style.display = missing ? "flex" : "none";
    if (missing) {
        ui.liveAria2Text.textContent = "aria2c is missing — downloads cannot start without it.";
        ui.liveAria2Button.disabled = false;
        ui.liveAria2Button.textContent = "Install aria2";
    }
}

async function installAria2() {
    const ui = state.ui;
    if (!ui?.liveAria2Button) return;
    ui.liveAria2Button.disabled = true;
    ui.liveAria2Button.textContent = "Installing…";
    ui.liveAria2Text.textContent = "Running apt-get install aria2. This takes a moment.";
    try {
        const res = await api.fetchApi("/hf-model-downloader/aria2", { method: "POST" });
        const data = await res.json();
        if (data.ok) {
            state.aria2 = { installed: true, checked: true, version: data.version || "" };
            setStatus(`aria2 ready${data.version ? ` — ${data.version}` : ""}.`, "success");
            renderAria2Warning();
            return;
        }
        ui.liveAria2Text.textContent = `Install failed: ${data.error || "unknown error"}`;
        ui.liveAria2Button.disabled = false;
        ui.liveAria2Button.textContent = "Retry";
    } catch (error) {
        ui.liveAria2Text.textContent = `Install failed: ${error.message || error}`;
        ui.liveAria2Button.disabled = false;
        ui.liveAria2Button.textContent = "Retry";
    }
}

async function runLiveSearch() {
    const ui = state.ui;
    if (!ui?.liveRepos) return;
    state.live.loading = true;
    ui.liveRepos.replaceChildren(el("div", "hfmd-live-empty", "Searching the Hub…"));

    const params = new URLSearchParams({
        q: state.live.query || "",
        sort: state.live.sort,
        pipeline: state.live.pipeline,
        limit: "40",
    });
    try {
        const res = await api.fetchApi(`/hf-model-downloader/search?${params}`);
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "search failed");
        state.live.results = data.results || [];
    } catch (error) {
        state.live.results = [];
        ui.liveRepos.replaceChildren(
            el("div", "hfmd-live-empty", `Search failed: ${error.message || error}`),
        );
        state.live.loading = false;
        return;
    }
    state.live.loading = false;
    renderLiveRepos();
}

function renderLiveRepos() {
    const ui = state.ui;
    if (!ui?.liveRepos) return;
    const rows = state.live.results;
    if (!rows.length) {
        ui.liveRepos.replaceChildren(
            el("div", "hfmd-live-empty", "Nothing matched. Try a shorter query."),
        );
        return;
    }

    const frag = document.createDocumentFragment();
    for (const repo of rows) {
        const card = el("div", "hfmd-live-repo");
        if (state.live.activeRepo === repo.repo_id) card.classList.add("is-active");
        card.appendChild(el("div", "hfmd-live-repo-id", repo.repo_id));
        const meta = el("div", "hfmd-live-repo-meta");
        meta.appendChild(el("span", "", `${formatCount(repo.downloads)} dl`));
        meta.appendChild(el("span", "", `${formatCount(repo.likes)} likes`));
        if (repo.pipeline) meta.appendChild(el("span", "hfmd-chip is-cyan", repo.pipeline));
        if (repo.gated) meta.appendChild(el("span", "hfmd-chip is-magenta", "gated"));
        card.appendChild(meta);
        card.addEventListener("click", () => openLiveRepo(repo.repo_id));
        frag.appendChild(card);
    }
    ui.liveRepos.replaceChildren(frag);
}

function formatCount(value) {
    const n = Number(value || 0);
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return String(n);
}

async function openLiveRepo(repoId) {
    const ui = state.ui;
    if (!ui?.liveFiles) return;
    state.live.activeRepo = repoId;
    state.live.files = [];
    state.live.selected.clear();
    renderLiveRepos();
    updateSelectionSummary();
    ui.liveFiles.replaceChildren(el("div", "hfmd-live-empty", `Reading ${repoId}…`));

    try {
        const res = await api.fetchApi(
            `/hf-model-downloader/repo?id=${encodeURIComponent(repoId)}`,
        );
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "could not read repo");
        state.live.files = data.files || [];
        renderLiveFiles(data);
    } catch (error) {
        ui.liveFiles.replaceChildren(
            el("div", "hfmd-live-empty", `Failed: ${error.message || error}`),
        );
    }
}

function renderLiveFiles(data) {
    const ui = state.ui;
    if (!ui?.liveFiles) return;
    const files = state.live.files;
    if (!files.length) {
        ui.liveFiles.replaceChildren(
            el(
                "div",
                "hfmd-live-empty",
                "No .safetensors / .ckpt / .gguf files in this repo. It may be diffusers-only or gated.",
            ),
        );
        return;
    }

    const frag = document.createDocumentFragment();
    const head = el("div", "hfmd-live-bar");
    head.appendChild(
        el(
            "div",
            "hfmd-live-hint",
            `${files.length} files · ${formatBytes(data.total_bytes)} total · ${data.repo.repo_id}`,
        ),
    );
    const selectAll = el("button", "hfmd-btn", "Select all");
    selectAll.addEventListener("click", () => {
        const everySelected = files.every((f) => state.live.selected.has(f.id));
        files.forEach((f) => {
            if (everySelected) state.live.selected.delete(f.id);
            else state.live.selected.add(f.id);
        });
        renderLiveFiles(data);
        updateSelectionSummary();
    });
    head.appendChild(selectAll);
    const openHub = el("button", "hfmd-btn", "Open on HF");
    openHub.addEventListener("click", () => window.open(data.readme_url, "_blank", "noopener"));
    head.appendChild(openHub);
    frag.appendChild(head);

    for (const file of files) {
        const row = el("div", "hfmd-live-file");
        if (file.installed) row.classList.add("is-installed");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = state.live.selected.has(file.id);
        box.addEventListener("change", () => {
            if (box.checked) state.live.selected.add(file.id);
            else state.live.selected.delete(file.id);
            updateSelectionSummary();
        });
        row.appendChild(box);

        const main = el("div", "hfmd-live-file-main");
        main.appendChild(el("div", "hfmd-live-file-name", file.path));
        const sub = el("div", "hfmd-live-file-sub");
        sub.appendChild(el("span", "", formatBytes(file.size)));
        sub.appendChild(el("span", "hfmd-chip is-cyan", file.category));
        if (file.family && file.family !== "MISC") {
            sub.appendChild(el("span", "hfmd-chip", file.family));
        }
        if (file.shard) sub.appendChild(el("span", "hfmd-chip is-magenta", "shard"));
        if (file.installed) sub.appendChild(el("span", "hfmd-chip is-green", "installed"));
        main.appendChild(sub);
        row.appendChild(main);
        frag.appendChild(row);
    }
    ui.liveFiles.replaceChildren(frag);
}

async function startLiveDownload() {
    const files = liveSelectedFiles();
    if (!files.length) {
        setStatus("Select at least one file first.", "error");
        return;
    }
    if (state.aria2.checked && !state.aria2.installed) {
        setStatus("aria2c is missing. Install it first.", "error");
        return;
    }

    setStatus(`Queueing ${files.length} file(s) from ${state.live.activeRepo}…`, "info");
    try {
        const res = await api.fetchApi("/hf-model-downloader/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                items: files.map((f) => ({
                    repo_id: f.repo_id,
                    repo_revision: f.repo_revision,
                    path: f.path,
                    size: f.size,
                    category: f.category,
                    family: f.family,
                    title: f.title,
                })),
            }),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || "download failed");
        state.jobId = data.job_id;
        state.live.selected.clear();
        updateSelectionSummary();
        setStatus(`Queued ${data.total} file(s).`, "success");
        setView("downloads");
        startPolling();
        startJobsPolling();
        fetchJobs();
    } catch (error) {
        setStatus(`Download failed: ${error.message || error}`, "error");
    }
}
