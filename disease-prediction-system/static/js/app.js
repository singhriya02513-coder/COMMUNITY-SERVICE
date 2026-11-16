(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const MIN_SYMPTOMS = Number(document.body.dataset.minSymptoms) || 3;
  const MAX_SYMPTOMS = Number(document.body.dataset.maxSymptoms) || 20;

  const state = {
    selected: new Set(),
    filter: "all",
    query: "",
    loading: false,
    hasResults: false,
  };

  const el = {
    header: document.getElementById("site-header"),
    navToggle: document.querySelector(".nav-toggle"),
    mobileNav: document.getElementById("mobile-nav"),
    search: document.getElementById("symptom-search"),
    searchClear: document.getElementById("search-clear"),
    filterRow: document.getElementById("filter-row"),
    symptomGroups: document.getElementById("symptom-groups"),
    groups: Array.from(document.querySelectorAll(".symptom-group")),
    emptySearch: document.getElementById("empty-search"),
    resetSearch: document.getElementById("reset-search"),
    example: document.getElementById("example-btn"),
    clear: document.getElementById("clear-btn"),
    count: document.getElementById("count-value"),
    selectedList: document.getElementById("selected-list"),
    summarySubtitle: document.getElementById("summary-subtitle"),
    progressText: document.getElementById("progress-text"),
    progressBar: document.getElementById("progress-bar"),
    progressFill: document.getElementById("progress-fill"),
    predict: document.getElementById("predict-btn"),
    predictLabel: document.getElementById("predict-label"),
    predictIcon: document.getElementById("predict-icon"),
    predictSpinner: document.getElementById("predict-spinner"),
    summaryError: document.getElementById("summary-error"),
    results: document.getElementById("results"),
    resultsSubtitle: document.getElementById("results-subtitle"),
    startOver: document.getElementById("start-over"),
    emergencyAlert: document.getElementById("emergency-alert"),
    emergencyText: document.getElementById("emergency-text"),
    lowConfidence: document.getElementById("low-confidence-alert"),
    topName: document.getElementById("top-name"),
    topDesc: document.getElementById("top-desc"),
    topSpecialist: document.getElementById("top-specialist"),
    topCoverage: document.getElementById("top-coverage"),
    topCoverageFill: document.getElementById("top-coverage-fill"),
    topMatchedCount: document.getElementById("top-matched-count"),
    topMatched: document.getElementById("top-matched"),
    topPrecautions: document.getElementById("top-precautions"),
    topRing: document.getElementById("top-ring"),
    topConfidence: document.getElementById("top-confidence"),
    otherResults: document.getElementById("other-results"),
    toast: document.getElementById("toast"),
  };

  const chips = Array.from(document.querySelectorAll(".symptom-chip"));
  const checkboxes = chips.map((chip) => chip.querySelector("input"));
  const checkboxByKey = new Map(checkboxes.map((box) => [box.value, box]));
  const labelByKey = new Map(checkboxes.map((box) => [box.value, box.dataset.label || box.value]));

  let toastTimer = 0;

  function icon(name) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("class", "icon");
    svg.setAttribute("aria-hidden", "true");
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `#icon-${name}`);
    svg.appendChild(use);
    return svg;
  }

  function percent(value) {
    const number = Number(value) || 0;
    return `${number.toFixed(1).replace(".0", "")}%`;
  }

  function showToast(message, isError = false) {
    window.clearTimeout(toastTimer);
    el.toast.textContent = message;
    el.toast.classList.toggle("is-error", isError);
    el.toast.classList.add("is-visible");
    toastTimer = window.setTimeout(() => {
      el.toast.classList.remove("is-visible");
    }, 3600);
  }

  function hideToast() {
    window.clearTimeout(toastTimer);
    el.toast.classList.remove("is-visible");
  }

  function setButtonLoading(loading) {
    state.loading = loading;
    el.predictLabel.textContent = loading ? "Analysing symptoms..." : "Predict condition";
    el.predictIcon.hidden = loading;
    el.predictSpinner.hidden = !loading;
    el.predict.disabled = loading || state.selected.size < MIN_SYMPTOMS;
  }

  function setError(message) {
    el.summaryError.textContent = message;
    el.summaryError.hidden = !message;
  }

  function invalidateResults() {
    if (!state.hasResults) return;
    state.hasResults = false;
    el.results.hidden = true;
  }

  function renderSelectedList() {
    el.selectedList.replaceChildren();

    if (state.selected.size === 0) {
      const empty = document.createElement("div");
      empty.className = "selected-empty";
      const orbit = document.createElement("span");
      orbit.className = "empty-orbit";
      orbit.appendChild(icon("plus"));
      const text = document.createElement("p");
      text.textContent = `Your selected symptoms will appear here. Add at least ${MIN_SYMPTOMS} to run the check.`;
      empty.append(orbit, text);
      el.selectedList.appendChild(empty);
      return;
    }

    const wrap = document.createElement("div");
    wrap.className = "selected-tags";

    state.selected.forEach((key) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "selected-tag";
      button.dataset.key = key;
      button.setAttribute("aria-label", `Remove ${labelByKey.get(key) || key}`);

      const text = document.createElement("span");
      text.textContent = labelByKey.get(key) || key;
      button.append(text, icon("x"));
      wrap.appendChild(button);
    });

    el.selectedList.appendChild(wrap);
  }

  function updateSelection() {
    const count = state.selected.size;

    el.count.textContent = String(count);
    el.progressText.textContent = count < MIN_SYMPTOMS ? `${count} / ${MIN_SYMPTOMS}` : `${count} selected`;
    el.progressBar.setAttribute("aria-valuenow", String(Math.min(count, MIN_SYMPTOMS)));

    const progress = Math.min(100, (count / MIN_SYMPTOMS) * 100);
    el.progressFill.style.width = `${progress}%`;
    el.progressBar.classList.toggle("is-complete", count >= MIN_SYMPTOMS);

    if (count === 0) {
      el.summarySubtitle.textContent = "Nothing selected yet";
    } else if (count < MIN_SYMPTOMS) {
      const remaining = MIN_SYMPTOMS - count;
      el.summarySubtitle.textContent = `${remaining} more symptom${remaining === 1 ? "" : "s"} needed`;
    } else {
      el.summarySubtitle.textContent = `${count} symptom${count === 1 ? "" : "s"} ready to analyse`;
    }

    renderSelectedList();
    el.predict.disabled = state.loading || count < MIN_SYMPTOMS;
    setError("");
    invalidateResults();
  }

  function clearAll() {
    state.selected.clear();
    chips.forEach((chip) => {
      chip.classList.remove("is-selected");
      const box = chip.querySelector("input");
      if (box) box.checked = false;
    });
    updateSelection();
  }

  function setSelection(keys) {
    clearAll();
    keys.forEach((key) => {
      const box = checkboxByKey.get(key);
      if (!box) return;
      box.checked = true;
      state.selected.add(key);
      box.closest(".symptom-chip").classList.add("is-selected");
    });
    updateSelection();
  }

  function applyFilters() {
    let visible = 0;

    el.groups.forEach((group) => {
      const matchesCategory = state.filter === "all" || state.filter === group.dataset.group;
      let groupVisible = 0;

      group.querySelectorAll(".symptom-chip").forEach((chip) => {
        const box = chip.querySelector("input");
        const haystack = `${box.dataset.label} ${box.value}`.toLowerCase();
        const matchesQuery = !state.query || haystack.includes(state.query);
        const show = matchesCategory && matchesQuery;
        chip.hidden = !show;
        if (show) groupVisible += 1;
      });

      group.hidden = groupVisible === 0;
      visible += groupVisible;
    });

    el.emptySearch.hidden = visible !== 0;
    el.symptomGroups.scrollTop = 0;
  }

  function clearSearch() {
    state.query = "";
    el.search.value = "";
    el.searchClear.hidden = true;
    applyFilters();
    el.search.focus();
  }

  function resetFilters() {
    state.filter = "all";
    el.filterRow.querySelectorAll(".filter-chip").forEach((button) => {
      const active = button.dataset.filter === "all";
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    state.query = "";
    el.search.value = "";
    el.searchClear.hidden = true;
    applyFilters();
  }

  function buildEmergencyText(symptoms) {
    const list = symptoms.join(", ");
    return `You selected ${list}. If these symptoms are severe, sudden or getting worse, contact emergency services or go to the nearest emergency department now.`;
  }

  function renderTopPrediction(top) {
    el.topName.textContent = top.name;
    el.topDesc.textContent = top.description || "No description available.";
    el.topSpecialist.textContent = top.specialist || "General Physician";

    const confidence = Math.min(100, Math.max(0, Number(top.confidence) || 0));
    el.topRing.style.setProperty("--value", String(confidence));
    el.topConfidence.textContent = percent(confidence);

    const coverage = Math.min(100, Math.max(0, Number(top.coverage) || 0));
    el.topCoverage.textContent = `${coverage}%`;
    el.topCoverageFill.style.width = `${coverage}%`;
    el.topMatchedCount.textContent = `${top.matched_count} of ${top.total_known_symptoms} recorded symptoms matched`;

    el.topMatched.replaceChildren();
    const matched = Array.isArray(top.matched_symptoms) ? top.matched_symptoms : [];
    if (matched.length === 0) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "No direct symptom overlap";
      el.topMatched.appendChild(tag);
    } else {
      matched.forEach((label) => {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = label;
        el.topMatched.appendChild(tag);
      });
    }

    el.topPrecautions.replaceChildren();
    (top.precautions || []).forEach((precaution) => {
      const item = document.createElement("li");
      item.textContent = precaution;
      el.topPrecautions.appendChild(item);
    });
  }

  function buildOtherCard(prediction, rank) {
    const card = document.createElement("article");
    card.className = "other-card panel";

    const top = document.createElement("div");
    top.className = "other-top";

    const rankLabel = document.createElement("span");
    rankLabel.className = "other-rank";
    rankLabel.textContent = `Possibility ${String(rank).padStart(2, "0")}`;

    const confidence = document.createElement("span");
    confidence.className = "other-confidence";
    confidence.textContent = percent(prediction.confidence);

    top.append(rankLabel, confidence);

    const title = document.createElement("h3");
    title.textContent = prediction.name;

    const description = document.createElement("p");
    description.className = "other-desc";
    description.textContent = prediction.description || "";

    const bar = document.createElement("div");
    bar.className = "other-bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.min(100, Math.max(0, Number(prediction.confidence) || 0))}%`;
    bar.appendChild(fill);

    const meta = document.createElement("div");
    meta.className = "other-meta";

    const matched = document.createElement("span");
    matched.append(
      icon("check"),
      document.createTextNode(
        `${prediction.matched_count || 0} of ${prediction.total_known_symptoms || 0} symptoms matched`
      )
    );

    const specialist = document.createElement("span");
    specialist.append(
      icon("stethoscope"),
      document.createTextNode(prediction.specialist || "General Physician")
    );

    meta.append(matched, specialist);
    card.append(top, title, description, bar, meta);
    return card;
  }

  function renderResults(data) {
    const predictions = Array.isArray(data.predictions) ? data.predictions : [];
    if (predictions.length === 0) {
      throw new Error("No predictions were returned.");
    }

    renderTopPrediction(predictions[0]);

    el.otherResults.replaceChildren();
    predictions.slice(1).forEach((prediction, index) => {
      el.otherResults.appendChild(buildOtherCard(prediction, index + 2));
    });

    const emergency = Array.isArray(data.emergency_symptoms) ? data.emergency_symptoms : [];
    const urgentCondition = Boolean(data.urgent_condition);
    el.emergencyAlert.hidden = emergency.length === 0 && !urgentCondition;
    if (emergency.length > 0) {
      el.emergencyText.textContent = buildEmergencyText(emergency);
    } else if (urgentCondition) {
      el.emergencyText.textContent =
        "The top prediction is a condition that needs urgent medical assessment. Contact a doctor or emergency services now rather than waiting.";
    }

    el.lowConfidence.hidden = !data.low_confidence;
    el.resultsSubtitle.textContent = `${data.symptom_count} symptom${
      data.symptom_count === 1 ? "" : "s"
    } analysed and ranked by the trained model.`;
  }

  async function runPrediction() {
    if (state.loading || state.selected.size < MIN_SYMPTOMS) return;

    hideToast();
    setError("");
    setButtonLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symptoms: Array.from(state.selected) }),
      });

      let data = {};
      try {
        data = await response.json();
      } catch (parseError) {
        throw new Error("The server returned an unreadable response.");
      }

      if (!response.ok) {
        throw new Error(data.error || "The prediction request failed.");
      }

      el.results.hidden = false;
      renderResults(data);
      state.hasResults = true;

      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        el.results.scrollIntoView({ block: "start" });
      } else {
        el.results.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Something went wrong. Please try again.";
      setError(message);
      showToast(message, true);
    } finally {
      setButtonLoading(false);
    }
  }

  chips.forEach((chip) => {
    const box = chip.querySelector("input");
    if (!box) return;

    box.addEventListener("change", () => {
      if (box.checked && state.selected.size >= MAX_SYMPTOMS) {
        box.checked = false;
        chip.classList.remove("is-selected");
        showToast(`You can select up to ${MAX_SYMPTOMS} symptoms at a time.`, true);
        return;
      }

      if (box.checked) {
        state.selected.add(box.value);
        chip.classList.add("is-selected");
      } else {
        state.selected.delete(box.value);
        chip.classList.remove("is-selected");
      }
      updateSelection();
    });
  });

  el.search.addEventListener("input", () => {
    state.query = el.search.value.trim().toLowerCase();
    el.searchClear.hidden = state.query.length === 0;
    applyFilters();
  });

  el.searchClear.addEventListener("click", clearSearch);
  el.resetSearch.addEventListener("click", clearSearch);

  el.filterRow.addEventListener("click", (event) => {
    const button = event.target.closest(".filter-chip");
    if (!button) return;
    state.filter = button.dataset.filter;
    el.filterRow.querySelectorAll(".filter-chip").forEach((chip) => {
      const active = chip === button;
      chip.classList.toggle("is-active", active);
      chip.setAttribute("aria-pressed", String(active));
    });
    applyFilters();
  });

  el.selectedList.addEventListener("click", (event) => {
    const button = event.target.closest(".selected-tag");
    if (!button) return;
    const key = button.dataset.key;
    const box = checkboxByKey.get(key);
    if (box) {
      box.checked = false;
      box.closest(".symptom-chip").classList.remove("is-selected");
    }
    state.selected.delete(key);
    updateSelection();
  });

  el.clear.addEventListener("click", () => {
    clearAll();
    showToast("All symptoms cleared.");
  });

  el.example.addEventListener("click", () => {
    const example = ["fever", "dry_cough", "fatigue", "headache", "sore_throat"];
    setSelection(example);
    resetFilters();
    setError("");
    showToast("Example symptoms loaded. Run the prediction to see a result.");
  });

  el.predict.addEventListener("click", runPrediction);

  el.startOver.addEventListener("click", () => {
    clearAll();
    resetFilters();
    setError("");
    document.getElementById("checker").scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    });
  });

  el.navToggle.addEventListener("click", () => {
    const open = el.navToggle.getAttribute("aria-expanded") === "true";
    el.navToggle.setAttribute("aria-expanded", String(!open));
    el.navToggle.setAttribute("aria-label", open ? "Open menu" : "Close menu");
    el.mobileNav.classList.toggle("is-open", !open);
  });

  el.mobileNav.addEventListener("click", (event) => {
    if (!event.target.closest("a")) return;
    el.navToggle.setAttribute("aria-expanded", "false");
    el.navToggle.setAttribute("aria-label", "Open menu");
    el.mobileNav.classList.remove("is-open");
  });

  document.addEventListener("click", (event) => {
    if (!el.mobileNav.classList.contains("is-open")) return;
    if (el.mobileNav.contains(event.target) || el.navToggle.contains(event.target)) return;
    el.navToggle.setAttribute("aria-expanded", "false");
    el.navToggle.setAttribute("aria-label", "Open menu");
    el.mobileNav.classList.remove("is-open");
  });

  const updateHeader = () => {
    el.header.classList.toggle("is-scrolled", window.scrollY > 8);
  };

  window.addEventListener("scroll", updateHeader, { passive: true });

  const inlineFeatures = document.querySelectorAll(".feature-track span");
  inlineFeatures.forEach((bar) => {
    const width = bar.style.width;
    bar.style.width = "0";
    window.requestAnimationFrame(() => {
      bar.style.width = width;
      bar.style.transition = "width 0.9s ease";
    });
  });

  updateSelection();
  applyFilters();
  updateHeader();
})();
