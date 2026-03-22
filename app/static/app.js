(function () {
  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderList(items) {
    if (!items || !items.length) {
      return "";
    }
    return "<ul>" + items.map((item) => `<li>${escapeHtml(item)}</li>`).join("") + "</ul>";
  }

  function renderMarkdownBlock(value, className) {
    const classes = ["render-markdown"];
    if (className) {
      classes.push(className);
    }
    return `<div class="${classes.join(" ")}">${escapeHtml(value || "")}</div>`;
  }

  function renderFallbackMarkdown(text) {
    return escapeHtml(text || "").replace(/\n/g, "<br>");
  }

  function normalizeBracketDisplayMath(raw) {
    return String(raw || "").replace(
      /(^|\n)\[\s*\n([\s\S]*?)\n\](?=\n|$)/g,
      function (_, prefix, content) {
        return `${prefix}$$\n${content}\n$$`;
      }
    );
  }

  function normalizeMathNotation(value) {
    return String(value || "")
      .replace(/（/g, "(")
      .replace(/）/g, ")")
      .replace(/，/g, ",")
      .replace(/；/g, ";")
      .replace(/：/g, ":")
      .replace(/【/g, "[")
      .replace(/】/g, "]");
  }

  function normalizeRichTextSource(raw) {
    const source = normalizeBracketDisplayMath(raw);
    const trimmed = source.trim();
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
      return source;
    }

    try {
      const payload = JSON.parse(trimmed);
      if (!payload || typeof payload !== "object") {
        return source;
      }
      if (!payload.final_answer && !payload.unresolved_disagreements && !payload.open_questions) {
        return source;
      }

      const blocks = [];
      if (payload.final_answer) {
        blocks.push(String(payload.final_answer));
      }
      if (Array.isArray(payload.unresolved_disagreements) && payload.unresolved_disagreements.length) {
        blocks.push("**Unresolved disagreements**\n" + payload.unresolved_disagreements.map((item) => `- ${item}`).join("\n"));
      }
      if (Array.isArray(payload.open_questions) && payload.open_questions.length) {
        blocks.push("**Open questions**\n" + payload.open_questions.map((item) => `- ${item}`).join("\n"));
      }
      return blocks.join("\n\n");
    } catch (error) {
      return source;
    }
  }

  function extractMathSegments(raw) {
    const segments = [];
    let text = normalizeBracketDisplayMath(raw);
    const patterns = [
      /\$\$[\s\S]+?\$\$/g,
      /\\\[[\s\S]+?\\\]/g,
      /\\\([\s\S]+?\\\)/g,
      /\$(?!\$)(?:\\.|[^$\n\\])+\$/g
    ];

    patterns.forEach((pattern) => {
      text = text.replace(pattern, function (match) {
        const token = `@@MATH_${segments.length}@@`;
        segments.push({ token, value: normalizeMathNotation(match) });
        return token;
      });
    });

    return { text, segments };
  }

  function restoreMathSegments(html, segments) {
    return segments.reduce((output, segment) => output.replaceAll(segment.token, segment.value), html);
  }

  function enhanceRichText(root) {
    const container = root || document;
    container.querySelectorAll(".render-markdown").forEach((node) => {
      if (node.dataset.rendered === "true") {
        return;
      }
      const raw = normalizeRichTextSource(node.textContent || "");
      const extracted = extractMathSegments(raw);
      let html = renderFallbackMarkdown(raw);
      if (window.marked && typeof window.marked.parse === "function") {
        try {
          html = window.marked.parse(extracted.text, { breaks: true, gfm: true });
        } catch (error) {
          console.error("Markdown render failed", error);
        }
      }
      if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
        html = window.DOMPurify.sanitize(html);
      }
      html = restoreMathSegments(html, extracted.segments);
      node.innerHTML = html;
      node.dataset.rendered = "true";
      node.classList.add("markdown-body");
      if (window.renderMathInElement) {
        try {
          window.renderMathInElement(node, {
            delimiters: [
              { left: "$$", right: "$$", display: true },
              { left: "\\[", right: "\\]", display: true },
              { left: "\\(", right: "\\)", display: false },
              { left: "$", right: "$", display: false }
            ],
            throwOnError: false
          });
        } catch (error) {
          console.error("LaTeX render failed", error);
        }
      }
    });
  }

  function parseRegistry() {
    const node = document.getElementById("registry-data");
    if (!node) {
      return null;
    }
    try {
      return JSON.parse(node.textContent || "{}");
    } catch (error) {
      console.error("Failed to parse registry data", error);
      return null;
    }
  }

  function discussionById(registry, discussionId) {
    return (registry?.discussions || []).find((discussion) => discussion.discussion_id === discussionId) || null;
  }

  function summarizePrompt(promptValue) {
    if (!promptValue) {
      return "Missing prompt";
    }
    if (promptValue.startsWith("file:")) {
      return promptValue;
    }
    return "inline prompt";
  }

  function splitModelRef(modelRef) {
    const parts = String(modelRef || "").split("/");
    if (parts.length < 2) {
      return { providerId: "unknown", modelName: "missing model" };
    }
    return {
      providerId: parts.shift() || "unknown",
      modelName: parts.join("/") || "missing model",
    };
  }

  function providerById(registry, providerId) {
    return (registry?.providers || []).find((provider) => provider.provider_id === providerId) || null;
  }

  function modelsForProvider(registry, providerId) {
    return (registry?.models || []).filter((model) => model.provider_id === providerId);
  }

  function providerOptionsHtml(registry, selectedProviderId) {
    return (registry?.providers || []).map((provider) => `
      <option value="${escapeHtml(provider.provider_id)}" ${provider.provider_id === selectedProviderId ? "selected" : ""}>
        ${escapeHtml(provider.display_name)}
      </option>
    `).join("");
  }

  function modelOptionsHtml(registry, selectedProviderId, selectedModelName) {
    const grouped = {};
    (registry?.models || []).forEach((model) => {
      if (selectedProviderId && model.provider_id !== selectedProviderId) {
        return;
      }
      grouped[model.provider_id] = grouped[model.provider_id] || [];
      grouped[model.provider_id].push(model);
    });

    const providerIds = Object.keys(grouped);
    if (!providerIds.length) {
      return selectedModelName
        ? `<option value="${escapeHtml(selectedModelName)}" selected>${escapeHtml(selectedModelName)}</option>`
        : `<option value="">No models configured in models.json</option>`;
    }

    const html = providerIds.map((providerId) => {
      const provider = providerById(registry, providerId);
      const options = grouped[providerId].map((model) => {
        const modelName = splitModelRef(model.model).modelName;
        const isSelected = modelName === selectedModelName;
        return `<option value="${escapeHtml(modelName)}" ${isSelected ? "selected" : ""}>${escapeHtml(model.label)} (${escapeHtml(modelName)})</option>`;
      }).join("");
      const label = provider ? provider.display_name : providerId;
      return selectedProviderId
        ? options
        : `<optgroup label="${escapeHtml(label)}">${options}</optgroup>`;
    }).join("");

    if (selectedModelName && !html.includes(`value="${escapeHtml(selectedModelName)}"`)) {
      return html + `<option value="${escapeHtml(selectedModelName)}" selected>${escapeHtml(selectedModelName)}</option>`;
    }
    return html;
  }

  function participantRowHtml(participant, index) {
    const modelBits = splitModelRef(participant.model);
    const promptSource = participant.prompt_source || participant.prompt || "";
    return `
      <article class="participant-row" data-index="${index}">
        <div class="participant-row-header">
          <h4>Participant ${index + 1}</h4>
          <button type="button" class="danger-link remove-participant-button">Remove</button>
        </div>
        <div class="participant-grid">
          <input type="hidden" name="participants-${index}-sort_order" value="${escapeHtml(participant.sort_order ?? index)}">
          <input type="hidden" name="participants-${index}-participant_id" value="${escapeHtml(participant.participant_id || `participant_${index + 1}`)}">
          <label>
            <span>Name</span>
            <input type="text" name="participants-${index}-name" value="${escapeHtml(participant.name || "")}">
          </label>
          <label>
            <span>Provider / platform</span>
            <select name="participants-${index}-provider_id" class="participant-provider-select">
              ${providerOptionsHtml(window.discussionRegistry, modelBits.providerId)}
            </select>
          </label>
          <label>
            <span>Model</span>
            <select name="participants-${index}-model_name" class="participant-model-select">
              ${modelOptionsHtml(window.discussionRegistry, modelBits.providerId, modelBits.modelName)}
            </select>
          </label>
          <label>
            <span>Role label</span>
            <input type="text" name="participants-${index}-role_label" value="${escapeHtml(participant.role_label || "")}" placeholder="Optional">
          </label>
          <label class="checkbox-label">
            <span>Enabled</span>
            <input type="checkbox" name="participants-${index}-enabled" value="true" ${participant.enabled === false ? "" : "checked"}>
          </label>
        </div>
        <label>
          <span>Prompt or file path</span>
          <textarea name="participants-${index}-prompt" rows="4" placeholder="Inline prompt text or file:dialog/general/answerer.txt">${escapeHtml(participant.prompt || "")}</textarea>
        </label>
        <p class="participant-meta">
          Provider: <strong>${escapeHtml(modelBits.providerId)}</strong>
          <span class="meta-divider">|</span>
          Model: <strong>${escapeHtml(modelBits.modelName)}</strong>
          <span class="meta-divider">|</span>
          Prompt source: <strong>${escapeHtml(summarizePrompt(promptSource))}</strong>
        </p>
      </article>
    `;
  }

  function syncParticipantMeta(row) {
    const providerSelect = row.querySelector(".participant-provider-select");
    const modelSelect = row.querySelector(".participant-model-select");
    const promptInput = row.querySelector('textarea[name$="-prompt"]');
    const meta = row.querySelector(".participant-meta");
    if (!providerSelect || !modelSelect || !meta || !promptInput) {
      return;
    }
    const provider = providerById(window.discussionRegistry, providerSelect.value);
    const modelBits = {
      providerId: providerSelect.value || "unknown",
      modelName: modelSelect.value || "missing model",
    };
    const promptSource = summarizePrompt(promptInput.value.trim());
    meta.innerHTML = `
      Provider: <strong>${escapeHtml(provider ? provider.display_name : modelBits.providerId)}</strong>
      <span class="meta-divider">|</span>
      Model: <strong>${escapeHtml(modelBits.modelName)}</strong>
      <span class="meta-divider">|</span>
      Prompt source: <strong>${escapeHtml(promptSource)}</strong>
    `;
  }

  function syncModelOptions(row, preserveSelection) {
    const providerSelect = row.querySelector(".participant-provider-select");
    const modelSelect = row.querySelector(".participant-model-select");
    if (!providerSelect || !modelSelect) {
      return;
    }
    const currentValue = preserveSelection ? modelSelect.value : "";
    modelSelect.innerHTML = modelOptionsHtml(window.discussionRegistry, providerSelect.value, currentValue);
    if (!modelSelect.value) {
      const firstEnabled = modelSelect.querySelector("option");
      if (firstEnabled) {
        modelSelect.value = firstEnabled.value;
      }
    }
  }

  function reindexParticipantRows(participantList) {
    const rows = Array.from(participantList.querySelectorAll(".participant-row"));
    rows.forEach((row, index) => {
      row.dataset.index = index;
      const title = row.querySelector(".participant-row-header h4");
      if (title) {
        title.textContent = `Participant ${index + 1}`;
      }
      row.querySelectorAll("[name]").forEach((element) => {
        element.name = element.name.replace(/participants-\d+-/, `participants-${index}-`);
      });
      const hiddenSort = row.querySelector(`input[name="participants-${index}-sort_order"]`);
      if (hiddenSort) {
        hiddenSort.value = String(index);
      }
      syncParticipantMeta(row);
    });
    const countField = document.getElementById("participant-count");
    if (countField) {
      countField.value = String(rows.length);
    }
  }

  function attachRowListeners(row) {
    const removeButton = row.querySelector(".remove-participant-button");
    const providerSelect = row.querySelector(".participant-provider-select");
    const modelSelect = row.querySelector(".participant-model-select");
    const promptInput = row.querySelector('textarea[name$="-prompt"]');
    if (removeButton) {
      removeButton.addEventListener("click", function () {
        const participantList = document.getElementById("participant-list");
        row.remove();
        if (participantList) {
          reindexParticipantRows(participantList);
        }
      });
    }
    if (providerSelect) {
      providerSelect.addEventListener("change", function () {
        syncModelOptions(row, false);
        syncParticipantMeta(row);
      });
    }
    [modelSelect, promptInput].forEach((element) => {
      if (element) {
        element.addEventListener("input", function () {
          syncParticipantMeta(row);
        });
        element.addEventListener("change", function () {
          syncParticipantMeta(row);
        });
      }
    });
    syncModelOptions(row, true);
    syncParticipantMeta(row);
  }

  function addParticipantRow(participant) {
    const participantList = document.getElementById("participant-list");
    if (!participantList) {
      return;
    }
    const index = participantList.querySelectorAll(".participant-row").length;
    participantList.insertAdjacentHTML("beforeend", participantRowHtml(participant, index));
    const row = participantList.querySelector(`.participant-row[data-index="${index}"]`);
    if (row) {
      attachRowListeners(row);
    }
    reindexParticipantRows(participantList);
  }

  function renderDiscussionSummary(discussion) {
    if (!discussion) {
      return "<span class=\"section-kicker\">Template</span><div>No template selected.</div>";
    }
    return `
      <span class="section-kicker">Template</span>
      <div><strong>${escapeHtml(discussion.display_name)}</strong></div>
      <p class="muted">${escapeHtml(discussion.description || "")}</p>
      <div class="catalog-meta">Max turn cycles: ${escapeHtml(discussion.max_turn_cycles)}</div>
      <ol class="participant-order">
        ${(discussion.participants || []).map((participant) => `
          <li>
            <strong>${escapeHtml(participant.name)}</strong>
            <span class="muted">${escapeHtml(participant.role_label || "Participant")} / ${escapeHtml(participant.model)} / ${escapeHtml(summarizePrompt(participant.prompt))}${participant.enabled === false ? " / disabled" : ""}</span>
          </li>
        `).join("")}
      </ol>
    `;
  }

  function updateSelectedDiscussionSummary(registry, discussionId) {
    const summary = document.getElementById("selected-discussion-summary");
    if (!summary) {
      return;
    }
    summary.innerHTML = renderDiscussionSummary(discussionById(registry, discussionId));
  }

  function applyDiscussionTemplate(registry, discussionId) {
    const participantList = document.getElementById("participant-list");
    if (!participantList) {
      return;
    }
    const discussion = discussionById(registry, discussionId);
    if (!discussion) {
      return;
    }
    participantList.innerHTML = "";
    (discussion.participants || []).forEach((participant) => addParticipantRow(participant));
    const maxTurnCycles = document.querySelector('input[name="max_turn_cycles"]');
    if (maxTurnCycles) {
      maxTurnCycles.value = String(discussion.max_turn_cycles || 1);
    }
    updateSelectedDiscussionSummary(registry, discussionId);
  }

  function renderTurn(turn) {
    const errorHtml = turn.error
      ? `<div class="error-banner compact-error"><strong>${escapeHtml(turn.error.code)}</strong><div>${escapeHtml(turn.error.message)}</div></div>`
      : "";
    const phaseLabel = turn.phase ? ` / ${escapeHtml(String(turn.phase).replaceAll("_", " "))}` : "";
    return `
      <article class="turn-card">
        <div class="turn-head">
          <div class="speaker-badge">${escapeHtml((turn.speaker || "?").slice(0, 1).toUpperCase())}</div>
          <div class="turn-main">
            <div class="turn-index">Turn ${escapeHtml(turn.turn_number)} / Cycle ${escapeHtml(turn.cycle_number)}${phaseLabel}</div>
            <h3>${escapeHtml(turn.speaker)}</h3>
            <p>${escapeHtml(turn.role_label || "Participant")}</p>
            <div class="turn-tags">
              <span class="meta-chip">${escapeHtml(turn.provider_display_name)}</span>
              <span class="meta-chip">${escapeHtml(turn.model_name)}</span>
              <span class="meta-chip meta-chip-subtle">${escapeHtml(turn.prompt_source)}</span>
            </div>
          </div>
          <span class="status-pill status-${escapeHtml(turn.status)}">${escapeHtml(turn.status)}</span>
        </div>
        <div class="turn-supporting">
          <span class="section-kicker">Route</span>
          <span>${escapeHtml(turn.model)}</span>
        </div>
        ${errorHtml}
        ${renderMarkdownBlock(turn.content || "Waiting for response...", "turn-content")}
      </article>
    `;
  }

  function renderSession(session) {
    const errors = session.errors?.length
      ? `
        <div class="error-banner">
          <strong>Errors</strong>
          <ul class="error-list">
            ${session.errors.map((error) => `<li>${escapeHtml(error.participant_id || "system")} / ${escapeHtml(error.provider_id)}: ${escapeHtml(error.message)}</li>`).join("")}
          </ul>
        </div>
      `
      : "";

    const participants = session.participants?.length
      ? `
        <div class="meta-block participant-summary">
          <span class="section-kicker">Session</span>
          <div>${escapeHtml(session.discussion_display_name || session.discussion_id || "Ad hoc discussion")}</div>
          <span class="section-kicker">Participants</span>
          <ul class="participant-roster">
            ${session.participants.map((participant) => `<li class="participant-roster-item"><strong>${escapeHtml(participant.name)}</strong><span>${escapeHtml(participant.role_label || "Participant")} / ${escapeHtml(participant.model_name || participant.model)}</span></li>`).join("")}
          </ul>
        </div>
      `
      : "";

    const gameState = session.game_state?.mode === "werewolf"
      ? `
        <div class="meta-block">
          <span class="section-kicker">Game State</span>
          <div>Phase: <strong>${escapeHtml(session.game_state.phase || "setup")}</strong></div>
          <div>Round: <strong>${escapeHtml(session.game_state.round_number || 0)}</strong></div>
          ${session.game_state.winner ? `<div>Winner: <strong>${escapeHtml(session.game_state.winner)}</strong></div>` : ""}
          <span class="section-kicker">Alive</span>
          <ul class="participant-roster">
            ${(session.game_state.alive_players || []).map((player) => `
              <li class="participant-roster-item"><strong>${escapeHtml(player.name)}</strong></li>
            `).join("")}
          </ul>
          ${(session.game_state.dead_players || []).length ? `
            <span class="section-kicker">Dead</span>
            <ul class="participant-roster">
              ${(session.game_state.dead_players || []).map((player) => `
                <li class="participant-roster-item">
                  <strong>${escapeHtml(player.name)}</strong>
                  <span>${escapeHtml(player.revealed_role || "Role hidden")}</span>
                </li>
              `).join("")}
            </ul>
          ` : ""}
          <span class="section-kicker">Latest Night</span>
          <div class="plain-text">${escapeHtml(session.game_state.last_night_summary || "No night result yet.")}</div>
          <span class="section-kicker">Latest Vote</span>
          <div class="plain-text">${escapeHtml(session.game_state.last_vote_summary || "No vote result yet.")}</div>
        </div>
      `
      : "";

    const moderatorView = session.game_state?.mode === "werewolf"
      ? `
        <div class="meta-block">
          <span class="section-kicker">Moderator View</span>
          <div class="muted">User-visible night truth. This panel is not shared back into the public AI transcript.</div>
          <span class="section-kicker">Role Map</span>
          <ul class="participant-roster">
            ${(session.game_state.moderator_view?.role_map || []).map((player) => `
              <li class="participant-roster-item">
                <strong>${escapeHtml(player.name)}</strong>
                <span>${escapeHtml(player.role)} / ${escapeHtml(player.team)} / ${escapeHtml(player.status)}</span>
              </li>
            `).join("")}
          </ul>
          <span class="section-kicker">Latest Night Truth</span>
          <div class="plain-text">${escapeHtml(session.game_state.moderator_view?.latest_night_private || "No night actions resolved yet.")}</div>
        </div>
      `
      : "";

    const timeline = session.turns?.length
      ? `<section class="timeline">${session.turns.map(renderTurn).join("")}</section>`
      : `
        <section class="timeline">
          <div class="empty-results">
            <p>No turns yet. The timeline will populate as participants speak.</p>
          </div>
        </section>
      `;

    const finalPanelTitle = session.game_state?.mode === "werewolf"
      ? "Final Game Summary"
      : "Final Synthesized Answer";

    return `
      <div class="panel results-panel">
        <div class="panel-header">
          <h2>Discussion Workbench</h2>
          <p>Use the sidebar for context and the main lane for the live discussion timeline.</p>
        </div>
        <div class="workbench-shell">
          <aside class="workbench-sidebar">
            <div class="stats-grid">
              <article class="stat-card">
                <span class="section-kicker">Turns</span>
                <strong>${escapeHtml(session.turns?.length || 0)}</strong>
              </article>
              <article class="stat-card">
                <span class="section-kicker">Cycles</span>
                <strong>${escapeHtml(session.settings?.max_turn_cycles || 0)}</strong>
              </article>
              <article class="stat-card">
                <span class="section-kicker">Status</span>
                <strong>${escapeHtml(session.status || "queued")}</strong>
              </article>
            </div>
            ${participants}
            ${gameState}
            ${moderatorView}
            ${session.unresolved_disagreements?.length ? `
              <div class="meta-block">
                <span class="section-kicker">Unresolved</span>
                ${renderList(session.unresolved_disagreements)}
              </div>
            ` : ""}
            ${session.open_questions?.length ? `
              <div class="meta-block">
                <span class="section-kicker">Open Questions</span>
                ${renderList(session.open_questions)}
              </div>
            ` : ""}
          </aside>
          <div class="workbench-main">
            <div class="question-panel">
              <span class="section-kicker">Original Question</span>
              ${renderMarkdownBlock(session.question || "")}
            </div>
            ${errors}
            ${timeline}
            <section class="final-answer-panel">
              <div class="section-header">
                <h3>${escapeHtml(finalPanelTitle)}</h3>
                <span class="status-pill status-${escapeHtml(session.status)}">${escapeHtml(session.status)}</span>
              </div>
              ${renderMarkdownBlock(session.final_answer || "No final answer yet.", "final-answer")}
            </section>
          </div>
        </div>
      </div>
    `;
  }

  function sessionRenderSignature(session) {
    const turnTail = (session.turns || []).slice(-1)[0];
    const errorTail = (session.errors || []).slice(-1)[0];
    return JSON.stringify({
      status: session.status || "",
      turnCount: session.turns?.length || 0,
      finalAnswer: session.final_answer || "",
      gameMode: session.game_state?.mode || "",
      gamePhase: session.game_state?.phase || "",
      gameRound: session.game_state?.round_number || 0,
      aliveCount: session.game_state?.alive_players?.length || 0,
      deadCount: session.game_state?.dead_players?.length || 0,
      gameWinner: session.game_state?.winner || "",
      unresolvedCount: session.unresolved_disagreements?.length || 0,
      openQuestionCount: session.open_questions?.length || 0,
      errorCount: session.errors?.length || 0,
      lastTurnStatus: turnTail?.status || "",
      lastTurnSpeaker: turnTail?.speaker || "",
      lastTurnContent: turnTail?.content || "",
      lastErrorMessage: errorTail?.message || "",
    });
  }

  async function pollSession(sessionId, shell) {
    let active = true;
    let previousSignature = "";
    while (active) {
      try {
        const response = await fetch(`/api/sessions/${sessionId}`, { headers: { Accept: "application/json" } });
        if (!response.ok) {
          break;
        }
        const session = await response.json();
        const signature = sessionRenderSignature(session);
        if (signature !== previousSignature) {
          shell.innerHTML = renderSession(session);
          enhanceRichText(shell);
          restoreWorkbenchScrollTarget();
          previousSignature = signature;
        }
        if (session.status !== "queued" && session.status !== "running") {
          active = false;
          break;
        }
      } catch (error) {
        console.error("Polling failed", error);
        break;
      }
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
  }

  function restoreWorkbenchScrollTarget() {
    const hash = window.location.hash || "";
    const pendingTarget = window.sessionStorage.getItem("discussion-scroll-target") || "";
    const targetId = hash === "#results-shell"
      ? "results-shell"
      : pendingTarget;
    if (!targetId) {
      return;
    }
    if (document.body.dataset.workbenchScrollDone === targetId) {
      if (pendingTarget) {
        window.sessionStorage.removeItem("discussion-scroll-target");
      }
      return;
    }
    const target = document.getElementById(targetId);
    if (!target) {
      return;
    }
    window.requestAnimationFrame(function () {
      target.scrollIntoView({ block: "start", behavior: "auto" });
      document.body.dataset.workbenchScrollDone = targetId;
    });
    if (pendingTarget) {
      window.sessionStorage.removeItem("discussion-scroll-target");
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    const registry = parseRegistry();
    window.discussionRegistry = registry;
    const participantList = document.getElementById("participant-list");
    if (registry && participantList) {
      participantList.querySelectorAll(".participant-row").forEach((row) => attachRowListeners(row));
      reindexParticipantRows(participantList);

      const discussionSelect = document.getElementById("discussion-select");
      if (discussionSelect) {
        updateSelectedDiscussionSummary(registry, discussionSelect.value);
        discussionSelect.addEventListener("change", function () {
          updateSelectedDiscussionSummary(registry, discussionSelect.value);
        });
      }

      const addButton = document.getElementById("add-participant-button");
      if (addButton) {
        addButton.addEventListener("click", function () {
          addParticipantRow({
            participant_id: `participant_${participantList.querySelectorAll(".participant-row").length + 1}`,
            name: `Participant ${participantList.querySelectorAll(".participant-row").length + 1}`,
            model: "openai_compatible/your-model",
            prompt: "file:dialog/general/answerer.txt",
            role_label: "",
            enabled: true,
            sort_order: participantList.querySelectorAll(".participant-row").length,
          });
        });
      }

      const applyDiscussionButton = document.getElementById("apply-discussion-button");
      if (applyDiscussionButton && discussionSelect) {
        applyDiscussionButton.addEventListener("click", function () {
          applyDiscussionTemplate(registry, discussionSelect.value);
        });
      }
    }

    const shell = document.getElementById("results-shell");
    const discussionForm = document.getElementById("discussion-form");
    if (discussionForm) {
      discussionForm.addEventListener("submit", function () {
        window.sessionStorage.setItem("discussion-scroll-target", "results-shell");
      });
    }
    if (shell) {
      const sessionId = shell.dataset.sessionId;
      const status = shell.dataset.sessionStatus;
      enhanceRichText(shell);
      restoreWorkbenchScrollTarget();
      if (sessionId && (status === "queued" || status === "running")) {
        pollSession(sessionId, shell);
      }
    }
  });
})();
