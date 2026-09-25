(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const state = { timer: null, clockTimer: null, refreshInFlight: null, snapshot: null, events: [], memories: [], auditTouched: false, audits: [], memoryPage: 1, memoryLimit: 50, memoryPageInfo: null, snapshotQuery: "", refreshedAt: 0, activeView: "overview" };
  const STATUS_LABELS = {
    received: "已接收", sent: "已送达", pending: "待处理", dispatched: "已派发",
    failed: "失败", unknown: "未知", processing: "处理中", cancelled: "已取消",
    active: "有效", candidate: "候选", rejected: "已拒绝", expired: "已过期",
  };
  const KIND_LABELS = { text: "文字", poke: "戳一戳", reaction: "回应", face: "QQ 表情", initiative: "主动消息" };
  const MEMORY_TYPE_LABELS = { preference: "偏好", agreement: "约定", correction: "纠正", self_expression: "角色自述" };
  const SOURCE_LABELS = { dialogue: "普通对话", interaction: "互动事件", initiative: "主动消息" };

  const text = (id, value) => {
    const node = $(id);
    node.textContent = value == null || value === "" ? "—" : String(value);
    return node;
  };

  const formatTime = (value) => {
    if (!value) return "—";
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { hour12: false });
  };

  const relativeTime = (value) => {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "时间不可解析";
    const seconds = Math.round((Date.now() - date.getTime()) / 1000);
    if (seconds < -30) return `${Math.abs(Math.round(seconds / 60))} 分钟后`;
    if (seconds < 10) return "刚刚";
    if (seconds < 60) return `${seconds} 秒前`;
    if (seconds < 3600) return `${Math.round(seconds / 60)} 分钟前`;
    if (seconds < 86400) return `${Math.round(seconds / 3600)} 小时前`;
    return `${Math.round(seconds / 86400)} 天前`;
  };

  const formatContext = (value) => value == null ? "—" : `${Number(value).toLocaleString("zh-CN")} token`;
  const label = (value, labels) => labels[value] || (value == null || value === "" ? "—" : String(value));
  const cssToken = (value) => String(value || "unknown").toLowerCase().replace(/[^a-z0-9_-]/g, "-");
  const compactIds = (value) => {
    if (!Array.isArray(value) || value.length === 0) return "无";
    const ids = value.slice(0, 2).join(", ");
    return `${value.length} 条 · ${ids}${value.length > 2 ? " …" : ""}`;
  };
  const shortId = (value) => (typeof value === "string" && value ? value.slice(0, 12) + "…" : "—");

  function showError(message) {
    $("error-banner").textContent = message;
    $("error-banner").hidden = false;
  }

  function clearError() { $("error-banner").hidden = true; }

  function setList(id, items, render) {
    const node = $(id);
    node.replaceChildren();
    if (!Array.isArray(items) || items.length === 0) {
      node.textContent = "暂无记录";
      node.classList.add("empty-list");
      return;
    }
    node.classList.remove("empty-list");
    items.forEach((item) => node.appendChild(render(item)));
  }

  function line(labelText, value, valueClass) {
    const node = document.createElement("div");
    node.className = "list-row";
    const key = document.createElement("span");
    key.textContent = labelText;
    const val = document.createElement("strong");
    val.textContent = value == null || value === "" ? "—" : String(value);
    if (valueClass) val.className = valueClass;
    node.append(key, val);
    return node;
  }

  function statusPill(value) {
    const node = document.createElement("span");
    node.className = `status-pill status-${cssToken(value)}`;
    node.textContent = label(value, STATUS_LABELS);
    return node;
  }

  function summarizeStatuses(items) {
    if (!Array.isArray(items) || items.length === 0) return "暂无记录";
    const counts = new Map();
    items.forEach((item) => {
      const amount = Number.isFinite(Number(item.count)) ? Number(item.count) : 1;
      counts.set(item.status, (counts.get(item.status) || 0) + amount);
    });
    const parts = [];
    if (counts.has("sent")) parts.push(`已送达 ${counts.get("sent")}`);
    if (counts.has("active")) parts.push(`有效 ${counts.get("active")}`);
    if (counts.has("pending")) parts.push(`待处理 ${counts.get("pending")}`);
    if (counts.has("dispatched")) parts.push(`已派发 ${counts.get("dispatched")}`);
    if (counts.has("failed")) parts.push(`失败 ${counts.get("failed")}`);
    if (counts.has("unknown")) parts.push(`未知 ${counts.get("unknown")}`);
    return parts.length ? parts.join(" · ") : `${items.length} 种状态`;
  }

  function countStatuses(items, statuses) {
    if (!Array.isArray(items)) return 0;
    return items.reduce((sum, item) => {
      if (!statuses.includes(item.status)) return sum;
      const count = Number(item.count);
      return sum + (Number.isFinite(count) ? count : 0);
    }, 0);
  }

  const levelLabel = (value, prefix) => value == null ? "—" : `${prefix || ""}${String(value).toUpperCase()}`;
  const certaintyLevel = (value) => ({unassessed: "C0", unsupported: "C0", ambiguous: "C1", explicit: "C2", confirmed: "C3"}[value] || (typeof value === "number" ? `C${value}` : value || "—"));
  const importanceLevel = (value) => value == null ? "—" : `I${value}`;
  const reasonLabel = (value) => ({explicit_user_statement: "用户明确表达", bilateral_agreement: "双方明确约定", later_user_confirmation: "用户后续确认", user_correction: "用户明确纠正", ambiguous_scope: "范围含糊", historical_event: "历史事件", contradicted_by_user: "被用户否定", expired_or_completed: "已完成或过期", unsupported_or_transient: "缺少支持或属短暂内容"}[value] || value || "未记录");
  const memoryOutcomeLabel = (value) => ({
    explicit_user_preference: "发现明确偏好", historical_episode: "发现历史情节",
    bilateral_bounded_agreement: "发现有期限的双方约定", existing_memory_review: "完成既有记忆审核",
    candidate_proposed: "提出记忆候选", review_proposed: "提出既有记忆审核",
    nothing_new: "没有新的持久记忆", temporary_scene_or_roleplay: "临时场景或角色扮演",
    ambiguous_scope: "适用范围含糊", missing_bilateral_acceptance: "缺少双方明确接受",
    insufficient_user_evidence: "用户证据不足", candidate_evidence_invalid: "候选证据无效",
  }[value] || value || "未记录");
  const recallScope = (item) => item.recall_scope || (item.temporal_scope === "always" ? "常驻" : item.temporal_scope === "topic" ? "按话题" : "—");

  const PRIVACY_LABELS = { adult: "含成人内容", intimate: "含亲密内容", ordinary: "日常" };
  const ACTOR_LABELS = { mumo: "他", qichi: "她", platform: "平台" };

  function fragmentSpan(item) {
    const start = formatTime(item.started_at_utc);
    const end = formatTime(item.ended_at_utc);
    return start === end ? start : start + " → " + String(end).slice(-5);
  }

  function renderFragments() {
    const node = $("fragment-timeline");
    node.replaceChildren();
    const fragments = Array.isArray(state.snapshot && state.snapshot.fragments) ? state.snapshot.fragments : [];
    $("fragments-count").textContent = fragments.length ? fragments.length + " 段" : "—";
    if (!fragments.length) {
      node.textContent = "暂无片段";
      node.classList.add("empty-list");
      return;
    }
    node.classList.remove("empty-list");
    fragments.forEach((item) => {
      const card = document.createElement("article");
      card.className = "fragment-card";
      const head = document.createElement("button");
      head.type = "button";
      head.className = "fragment-head";
      head.setAttribute("aria-expanded", "false");
      const when = document.createElement("strong");
      when.textContent = fragmentSpan(item);
      const meta = document.createElement("span");
      meta.className = "fragment-meta";
      const privacies = Object.entries(item.privacy_counts || {})
        .filter(([, count]) => count > 0)
        .map(([key, count]) => label(key, PRIVACY_LABELS) + " " + count)
        .join(" · ");
      meta.textContent = item.detail_count + " 条原文" + (privacies ? " · " + privacies : "");
      head.append(when, meta);
      const body = document.createElement("div");
      body.className = "fragment-body";
      body.hidden = true;
      card.append(head, body);
      head.addEventListener("click", () => { void toggleFragment(item, body, head); });
      node.append(card);
    });
  }

  async function toggleFragment(item, body, head) {
    const willOpen = body.hidden;
    body.hidden = !willOpen;
    head.setAttribute("aria-expanded", willOpen ? "true" : "false");
    if (!willOpen || body.dataset.loaded === "1") return;
    body.textContent = "读取中…";
    try {
      const payload = await getJson("/api/fragment?id=" + encodeURIComponent(item.fragment_id));
      body.dataset.loaded = "1";
      renderFragmentDetails(body, payload);
    } catch (error) {
      body.textContent = "读取失败：" + (error && error.message ? error.message : "未知错误");
    }
  }

  function renderFragmentDetails(body, payload) {
    body.replaceChildren();
    if (!payload || payload.ok !== true) {
      body.textContent = "没有读到这一段的原文：" + ((payload && payload.reason) || "未知原因");
      return;
    }
    const details = Array.isArray(payload.details) ? payload.details : [];
    if (!details.length) {
      body.textContent = "这一段没有留存逐条原文。";
      return;
    }
    const list = document.createElement("ol");
    list.className = "fragment-lines";
    const adults = details.filter((line) => line.privacy_class === "adult");
    let revealed = false;
    const draw = () => {
      list.replaceChildren();
      details.forEach((line) => {
        const adult = line.privacy_class === "adult";
        const item = document.createElement("li");
        item.className = adult ? "fragment-line fragment-line-adult" : "fragment-line";
        const meta = document.createElement("span");
        meta.className = "fragment-line-meta";
        meta.textContent = [line.ordinal, label(line.actor, ACTOR_LABELS), formatTime(line.occurred_at_utc),
          label(line.privacy_class, PRIVACY_LABELS)].join(" · ");
        const text = document.createElement("p");
        text.className = "fragment-line-text";
        text.textContent = adult && !revealed ? "（成人内容，默认折叠）" : line.exact_quote;
        item.append(meta, text);
        list.append(item);
      });
    };
    draw();
    if (adults.length) {
      const reveal = document.createElement("button");
      reveal.type = "button";
      reveal.className = "fragment-reveal";
      const caption = () => (revealed ? "收起成人原文" : "展开 " + adults.length + " 条成人原文");
      reveal.textContent = caption();
      reveal.addEventListener("click", () => { revealed = !revealed; reveal.textContent = caption(); draw(); });
      body.append(reveal);
    }
    body.append(list);
  }

  function renderRecall(node, payload) {
    node.replaceChildren();
    if (!payload || payload.ok !== true) {
      node.textContent = "无法复算：" + ((payload && payload.reason) || "未知原因");
      return;
    }
    const verdict = document.createElement("p");
    verdict.className = payload.explicit_request === true ? "recall-verdict recall-hit" : "recall-verdict";
    verdict.textContent = payload.explicit_request === true
      ? "词表命中：这一句算明确要求回顾，明细原文会被展开。"
      : "词表未命中：按当前规则这一句不会展开明细原文（只有原生引用或冻结词表才会）。";
    node.append(verdict);
    const describe = (items, withScore) => (Array.isArray(items) && items.length
      ? items.map((entry) => {
        const span = fragmentSpan(entry);
        const score = withScore && entry.score != null ? "（窗口命中 " + entry.score + "）" : "";
        return span + " seq " + entry.start_sequence + score;
      }).join("；")
      : "无");
    const rows = [
      ["解析出的日期", (payload.dates || []).join("、") || "无"],
      ["按日期命中", describe(payload.by_date, false)],
      ["按内容词命中", describe(payload.by_content, true)],
      ["兜底：最近一次", payload.newest ? describe([payload.newest], false) : "无片段"],
      ["词法窗口", (payload.windows || []).join(" / ") || "无"],
    ];
    const facts = document.createElement("dl");
    facts.className = "recall-facts";
    rows.forEach(([term, value]) => {
      const wrapper = document.createElement("div");
      const dt = document.createElement("dt");
      dt.textContent = term;
      const dd = document.createElement("dd");
      dd.textContent = value;
      wrapper.append(dt, dd);
      facts.append(wrapper);
    });
    node.append(facts);
  }

  async function runRecallExplanation(text) {
    const node = $("recall-result");
    node.className = "recall-result";
    node.textContent = "复算中…";
    try {
      renderRecall(node, await getJson("/api/recall?q=" + encodeURIComponent(text)));
    } catch (error) {
      node.textContent = "复算失败：" + (error && error.message ? error.message : "未知错误");
    }
  }

  function formatBytes(value) {
    if (typeof value !== "number" || !isFinite(value) || value <= 0) return "—";
    const units = ["B", "KB", "MB", "GB"];
    let size = value;
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit += 1; }
    return (unit === 0 ? size : size.toFixed(1)) + " " + units[unit];
  }

  function renderOperations() {
    const operations = (state.snapshot && state.snapshot.operations) || null;
    if (!operations) {
      text("ops-build-status", "无数据");
      return;
    }
    const build = operations.build || {};
    text("ops-build-code", shortId(build.code_build_id));
    text("ops-build-marker", shortId(build.marker_build_id));
    const badge = $("ops-build-status");
    if (build.matches === true) {
      badge.textContent = "一致";
      badge.className = "badge ok";
      text("ops-build-note", "运行中的进程与磁盘上的代码是同一份构建。");
    } else if (build.code_build_id) {
      badge.textContent = "需要重启";
      badge.className = "badge bad";
      text("ops-build-note", "磁盘代码已经变了，运行中的进程还是旧构建；重启后才会对齐。");
    } else {
      badge.textContent = "无法计算";
      badge.className = "badge";
      text("ops-build-note", "当前代码根下找不到运行时源码，无法计算指纹。");
    }
    const database = operations.database || {};
    text("ops-db-name", database.name || "—");
    text("ops-db-schema", database.schema_version == null ? "—" : "v" + database.schema_version);
    text("ops-db-size", formatBytes(database.bytes));
    text("ops-db-wal", formatBytes(database.wal_bytes));
    setList("ops-backups", operations.backups, (item) =>
      line(item.name, formatTime(item.modified_at_utc) + " · " + formatBytes(item.bytes)));
    const watermarks = operations.watermarks || {};
    setList(
      "ops-watermarks",
      Object.entries(watermarks).map(([key, value]) => ({ key, value })),
      (item) => line(item.key, item.value == null ? "—" : String(item.value)),
    );
  }

  const describeMemoryReason = (reason) => {
    if (typeof reason !== "string" || !reason) return "";
    const overlap = reason.match(/^trigram_overlap:fact=(\d+);quote=(\d+)$/);
    if (overlap) {
      return "命中方式：你这句和它引用的原话有 " + overlap[2] + " 处三字重合（条目文字 " + overlap[1] + " 处）";
    }
    return "命中方式：" + reason;
  };

  function memoryList(items, emptyText, withReason, reasons, scores) {
    const box = document.createElement("div");
    box.className = "recall-memories";
    if (!Array.isArray(items) || !items.length) {
      box.textContent = emptyText;
      return box;
    }
    items.forEach((entry) => {
      const row = document.createElement("div");
      row.className = "recall-memory";
      const head = document.createElement("span");
      head.className = "recall-memory-head";
      head.textContent = entry.type + " · " + label(entry.status, STATUS_LABELS);
      const fact = document.createElement("p");
      fact.className = "recall-memory-fact";
      fact.textContent = entry.normalized_fact || "—";
      row.append(head, fact);
      if (withReason && entry.memory_id) {
        const bits = [];
        const reason = (reasons || {})[entry.memory_id];
        if (reason) bits.push(describeMemoryReason(reason));
        const score = (scores || {})[entry.memory_id];
        if (score != null) bits.push("相关度 " + score);
        if (bits.length) {
          const why = document.createElement("small");
          why.className = "recall-memory-why";
          why.textContent = bits.join(" · ");
          row.appendChild(why);
        }
      }
      box.append(row);
    });
    return box;
  }

  function renderMemoryRecall(item, details) {
    const wrap = document.createElement("section");
    wrap.className = "recall-block";
    const heading = document.createElement("p");
    heading.className = "recall-block-heading";
    heading.textContent = "她这一轮想起了什么";
    wrap.append(heading);
    // 2026-09-22：工作集也要显示。它每轮都注入，只是按相关度排序、超预算时才丢尾部的条目，
    // 所以它不是「按话题想起的」，而是「一直在场但会被压缩」。之前面板漏了它，
    // 用户看到的常驻内容比实际少得多。
    const groups = [
      ["每一轮都在的记忆（约定与纠正，不需要触发）", item.relationship_refs, "无", false],
      ["一直在场的工作集（偏好与近期经历，按相关度排序、超预算才丢尾部）",
        item.working_set_refs, "这一轮工作集是空的", false],
      ["因为你这句里出现了相关词才想起的", item.retrieved_refs || item.candidate_refs,
        "这一轮没有按话题想起别的记忆", true],
    ];
    groups.forEach(([title, items, empty, withReason]) => {
      const block = document.createElement("div");
      block.className = "recall-group";
      const term = document.createElement("h4");
      term.textContent = title;
      // 命中理由与相关度在上下文轨迹里（生成元数据只带 id 列表）。
      block.append(term, memoryList(
        items, empty, withReason,
        (details && details.memory_reasons) || item.memory_reasons,
        (details && details.memory_scores) || item.memory_scores,
      ));
      wrap.append(block);
    });
    wrap.append(renderDetailRecall(item, details));
    return wrap;
  }

  function renderDetailRecall(item, details) {
    const block = document.createElement("div");
    block.className = "recall-group";
    const term = document.createElement("h4");
    term.textContent = "当时说过的话（明细原文）";
    block.append(term);
    const count = details ? details.memory_detail_count : null;
    const fragments = (details && Array.isArray(details.memory_detail_fragments))
      ? details.memory_detail_fragments : [];
    if (!count || !fragments.length) {
      const note = document.createElement("p");
      note.className = "recall-note";
      note.textContent = "这一轮没有把原话摊开给她——她能看到的只有片段索引（时间、级别、条数）。" +
        "只有你引用某条消息、或明确说「细说 / 回忆一下」时，原话才会展开给她。";
      block.append(note);
      return block;
    }
    const why = document.createElement("p");
    why.className = "recall-note";
    why.textContent = "这一轮展开了 " + count + " 条原话（" +
      label(details.memory_detail_reason, DETAIL_REASON_LABELS) + "）。";
    block.append(why, detailLineButton(fragments[0]));
    return block;
  }

  function detailLineButton(fragmentId) {
    const holder = document.createElement("div");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "fragment-reveal";
    button.textContent = "读这一段的原话";
    const body = document.createElement("div");
    body.className = "fragment-body";
    body.hidden = true;
    button.addEventListener("click", async () => {
      body.hidden = !body.hidden;
      if (body.hidden || body.dataset.loaded === "1") return;
      body.textContent = "读取中…";
      try {
        const payload = await getJson("/api/fragment?id=" + encodeURIComponent(fragmentId));
        body.dataset.loaded = "1";
        renderFragmentDetails(body, payload);
      } catch (error) {
        body.textContent = "读取失败：" + (error && error.message ? error.message : "未知错误");
      }
    });
    holder.append(button, body);
    return holder;
  }

  function renderMemories() {
    const node = $("memories-list");
    const filter = $("memory-status-filter").value;
    const query = $("memory-search").value.trim().toLowerCase();
    const records = state.memories.filter((item) => {
      if (filter === "active" && item.status !== "active") return false;
      if (filter === "candidate" && item.status !== "candidate") return false;
      if (filter === "inactive" && !["superseded", "rejected", "expired"].includes(item.status)) return false;
      if (!query) return true;
      const evidence = Array.isArray(item.evidence) ? item.evidence : [];
      return [item.memory_id, item.type, item.normalized_fact, item.modality, item.status, ...evidence.flatMap((entry) => [entry.event_id, entry.exact_quote, entry.actor])]
        .some((value) => String(value || "").toLowerCase().includes(query));
    });
    node.replaceChildren();
    $("memory-detail-count").textContent = `${records.length} / ${state.memories.length} 条`;
    if (!records.length) {
      node.textContent = state.memories.length ? "没有匹配的记忆" : "暂无记忆明细";
      node.classList.add("empty-list");
      return;
    }
    node.classList.remove("empty-list");
    records.forEach((item, index) => {
      const button = document.createElement("button");
      button.type = "button"; button.className = "memory-master-item";
      button.dataset.memoryId = item.memory_id || "";
      button.append(statusPill(item.status), document.createTextNode(` ${label(item.type, MEMORY_TYPE_LABELS)} · ${certaintyLevel(item.certainty)} · ${importanceLevel(item.importance)}`));
      const fact = document.createElement("span"); fact.className = "memory-master-fact"; fact.textContent = item.normalized_fact || "—"; button.appendChild(fact);
      button.addEventListener("click", () => selectMemory(item));
      node.appendChild(button);
      if (index === 0) selectMemory(item);
    });
  }

  function selectMemory(item) {
    const node = $("memory-selected-detail"); node.replaceChildren(); node.classList.remove("empty-list");
    document.querySelectorAll(".memory-master-item").forEach((button) => button.classList.toggle("selected", button.dataset.memoryId === (item.memory_id || "")));
    const heading = document.createElement("div"); heading.className = "memory-detail-heading";
    const title = document.createElement("strong"); title.textContent = item.normalized_fact || "—";
    heading.append(title, statusPill(item.status)); node.appendChild(heading);
    const facts = document.createElement("dl"); facts.className = "memory-facts-grid";
    const add = (name, value) => { const wrap = document.createElement("div"); const dt = document.createElement("dt"); dt.textContent = name; const dd = document.createElement("dd"); dd.textContent = value == null || value === "" ? "—" : String(value); wrap.append(dt, dd); facts.appendChild(wrap); };
    add("certainty", certaintyLevel(item.certainty)); add("importance", importanceLevel(item.importance)); add("时间范围", item.temporal_scope); add("派生 recall_scope", recallScope(item)); add("原因码说明", `${item.assessment_reason_code || "—"} · ${reasonLabel(item.assessment_reason_code)}`); add("assessed_at", formatTime(item.assessed_at || item.assessed_at_utc)); add("替代 supersedes", item.supersedes || item.supersedes_id); add("记忆 ID", item.memory_id);
    node.appendChild(facts);
    const evidence = Array.isArray(item.evidence) ? [...item.evidence].sort((a, b) => String(a.occurred_at_utc || "").localeCompare(String(b.occurred_at_utc || ""))) : [];
    const evidenceBox = document.createElement("div"); evidenceBox.className = "memory-evidence"; const labelNode = document.createElement("span"); labelNode.className = "memory-evidence-label"; labelNode.textContent = "证据时间线"; evidenceBox.appendChild(labelNode);
    evidence.forEach((entry) => { const quote = document.createElement("p"); quote.textContent = `${entry.actor || "未知"} · ${entry.event_id || "—"} · #${entry.sequence == null ? "—" : entry.sequence} · ${entry.occurred_at_utc || "—"} · ${entry.role || entry.evidence_role || "—"} · ${entry.exact_quote || "—"}`; evidenceBox.appendChild(quote); });
    node.appendChild(evidenceBox);
    renderPublicEvents(node, "审核事件", item.audit_events, ["audit_event_id", "action", "before_status", "after_status", "assessment_reason_code", "occurred_at_utc"]);
    renderPublicEvents(node, "确认展示", item.confirmation_presentations, ["presentation_id", "fragment_key", "trigger_event_id", "context_version", "presented_at_utc"]);
  }

  function renderPublicEvents(parent, headingText, entries, fields) {
    if (!Array.isArray(entries) || entries.length === 0) return;
    const box = document.createElement("div"); box.className = "memory-public-events";
    const heading = document.createElement("span"); heading.className = "memory-evidence-label"; heading.textContent = headingText; box.appendChild(heading);
    entries.forEach((entry) => {
      const row = document.createElement("p");
      row.textContent = fields.map((field) => `${field}=${entry[field] == null || entry[field] === "" ? "—" : entry[field]}`).join(" · ");
      box.appendChild(row);
    });
    parent.appendChild(box);
  }

  const CONTEXT_CATEGORY_LABELS = {
    role_core: "角色核心", runtime_facts: "事实信封", current_input: "当前输入",
    recent_history: "最近对话", earlier_history: "更早历史", memory_working_set: "记忆工作集",
    memory_evidence: "记忆证据", relationship_state: "关系状态",
    memory_index: "片段索引", memory_details: "明细原文", direct_quotes: "直接引用",
    memory_footprint: "最近原文", time_gaps: "时间断层", history: "历史",
  };
  const DETAIL_REASON_LABELS = {
    day: "按你点名的日期定位", quote: "按你引用的那条定位", content: "按话里的内容词定位",
    recent_guess: "兜底：最近一次（已标注为推定）", none: "本轮未展开明细原文",
  };

  function phaseDetails(item, phaseName) {
    const phases = Array.isArray(item.phases) ? item.phases : [];
    const direct = phases.find((entry) => entry && entry.phase === phaseName);
    if (direct && direct.details) return direct.details;
    // 审计条目只带发送侧元数据；上下文剖面与生成用量在技术轨迹里（同一 trigger_event_id）。
    const traces = Array.isArray(state.snapshot && state.snapshot.traces) ? state.snapshot.traces : [];
    const match = traces.find((trace) => trace && trace.trigger_event_id && trace.trigger_event_id === item.trigger_event_id);
    const phase = match && Array.isArray(match.phases)
      ? match.phases.find((entry) => entry && entry.phase === phaseName)
      : null;
    return phase && phase.details ? phase.details : null;
  }

  function contextPhaseDetails(item) {
    return phaseDetails(item, "context");
  }

  // 供应商前缀缓存的命中额度：只认从第一个 token 起完全一致的前缀单元，所以这一行
  // 直接反映版式好不好（2026-09-12 实测：钟在第二行时只有 6.6%）。
  function cacheSummary(item) {
    const details = phaseDetails(item, "generation");
    if (!details || details.cache_hit_tokens == null || details.input_tokens == null) return null;
    const hit = Number(details.cache_hit_tokens) || 0;
    const total = Number(details.input_tokens) || 0;
    if (!total) return null;
    return `${hit} / ${total} token（${Math.round((hit / total) * 100)}%）`;
  }

  // 联网（2026-09-14）：这一轮查了什么、成没成、资料多大。面板只报事实，不展示资料正文。
  function netSummary(item) {
    const details = phaseDetails(item, "generation");
    if (!details || !details.tool_name) return null;
    const name = details.tool_name === "image_search" ? "以图搜图" : "文本检索";
    const query = details.tool_query ? "「" + details.tool_query + "」" : "";
    const state = details.tool_ok === false ? "失败（" + (details.tool_degraded || "未知") + "）" : "成功";
    const elapsed = details.tool_elapsed_ms == null ? "" : " · " + (Number(details.tool_elapsed_ms) / 1000).toFixed(1) + "s";
    const chars = details.tool_result_chars == null ? "" : " · 资料 " + details.tool_result_chars + " 字";
    return name + query + " · " + state + elapsed + chars;
  }

  function carriedImagesSummary(details) {
    if (!details || !details.images_carried) return null;
    return "沿用上一轮 " + details.images_carried + " 张（宽限窗口）";
  }

  function renderContextProfile(details) {
    const wrap = document.createElement("div");
    wrap.className = "ctx-profile";
    const heading = document.createElement("p");
    heading.className = "ctx-heading";
    const chip = document.createElement("span");
    const reason = details ? details.memory_detail_reason : null;
    chip.className = reason === "recent_guess" ? "ctx-chip ctx-chip-guess" : "ctx-chip";
    chip.textContent = reason ? label(reason, DETAIL_REASON_LABELS) : "历史轮次（未记录定位方式）";
    heading.append(chip);
    if (details && details.memory_detail_count != null) {
      const count = document.createElement("span");
      count.className = "muted";
      count.textContent = "明细 " + details.memory_detail_count + " 条";
      heading.append(count);
    }
    const cats = Object.entries((details && details.category_tokens) || {})
      .map(([key, value]) => [key, Number(value) || 0])
      .filter(([, value]) => value > 0)
      .sort((a, b) => b[1] - a[1]);
    const total = cats.reduce((sum, [, value]) => sum + value, 0);
    const totalNode = document.createElement("span");
    totalNode.className = "muted";
    totalNode.textContent = total ? "输入合计 " + total + " token" : "本轮无上下文记录";
    heading.append(totalNode);
    wrap.append(heading);
    if (!total) return wrap;
    const bar = document.createElement("div");
    bar.className = "ctx-bar";
    cats.forEach(([key, value]) => {
      const seg = document.createElement("span");
      seg.className = "ctx-seg ctx-seg-" + cssToken(key);
      seg.style.flexGrow = String(value);
      seg.title = label(key, CONTEXT_CATEGORY_LABELS) + " " + value + " token";
      bar.append(seg);
    });
    const legend = document.createElement("ul");
    legend.className = "ctx-legend";
    cats.forEach(([key, value]) => {
      const li = document.createElement("li");
      li.className = "ctx-item ctx-item-" + cssToken(key);
      const name = document.createElement("span");
      name.className = "ctx-name";
      name.textContent = label(key, CONTEXT_CATEGORY_LABELS);
      const num = document.createElement("span");
      num.className = "ctx-value";
      num.textContent = String(value);
      li.append(name, num);
      legend.append(li);
    });
    wrap.append(bar, legend);
    return wrap;
  }

  function renderResponseAudit() {
    const node = $("response-audit-list");
    node.replaceChildren();
    $("response-audit-count").textContent = `${state.audits.length} 条`;
    if (!state.audits.length) {
      node.textContent = "暂无可追溯回复";
      node.classList.add("empty-list");
      return;
    }
    node.classList.remove("empty-list");
    state.audits.forEach((item, index) => {
      const article = document.createElement("details");
      article.className = `audit-entry audit-${cssToken(item.status)}`;
      // 打开对话页先看最近一轮：这条默认展开，用户折叠后不再自动弹开。
      article.open = index === 0 && state.activeView === "trace" && state.auditTouched !== true;
      article.addEventListener("toggle", () => { state.auditTouched = true; });
      const summary = document.createElement("summary");
      summary.className = "audit-heading";
      const title = document.createElement("strong");
      title.textContent = `Q${item.sequence == null ? "—" : item.sequence}`;
      const time = document.createElement("time");
      time.dateTime = item.occurred_at_utc || "";
      time.textContent = formatTime(item.occurred_at_utc);
      const excerpt = document.createElement("span");
      excerpt.className = "audit-excerpt";
      excerpt.textContent = String(item.text || "—").replace(/\s+/g, " ").slice(0, 110);
      summary.append(title, time, statusPill(item.status), excerpt);
      const body = document.createElement("p");
      body.className = "audit-text";
      body.textContent = item.text || "—";
      const trace = document.createElement("ol");
      trace.className = "audit-trace";
      const addTrace = (labelText, value) => {
        const step = document.createElement("li");
        const term = document.createElement("strong");
        term.textContent = labelText;
        const detail = document.createElement("span");
        detail.textContent = value == null || value === "" ? "未记录" : String(value);
        step.append(term, detail);
        trace.appendChild(step);
      };
      const triggerPrefix = item.trigger_direction === "internal" ? "I" : "M";
      const trigger = item.trigger_event_id
        ? `${triggerPrefix}${item.trigger_sequence == null ? "—" : item.trigger_sequence} · ${item.trigger_kind || "事件"}`
        : "无持久触发证据";
      const relationCount = Array.isArray(item.relationship_memory_ids) ? item.relationship_memory_ids.length : 0;
      const retrievedCount = Array.isArray(item.retrieved_memory_ids) ? item.retrieved_memory_ids.length : 0;
      const context = [
        item.context_version == null ? null : `上下文 v${item.context_version}`,
        item.history_event_count == null ? null : `历史 ${item.history_event_count} 条`,
        `关系记忆 ${relationCount} 条`,
        `按话题召回 ${retrievedCount} 条`,
        item.quoted_event_id ? `钉入引用 ${item.quoted_event_id}` : "无引用输入",
      ].filter(Boolean).join(" · ");
      const contextDetails = contextPhaseDetails(item);
      const profile = renderContextProfile(contextDetails);
      const recall = renderMemoryRecall(item, contextDetails);
      addTrace("触发", trigger);
      addTrace("输入证据", context);
      const memoryFacts = (items) => Array.isArray(items) && items.length
        ? items.map((entry) => `${entry.type || "记忆"}：${entry.normalized_fact || "—"}（${label(entry.status, STATUS_LABELS)}）`).join("；")
        : "无持久记忆依据";
      addTrace("关系记忆内容", memoryFacts(item.relationship_refs));
      addTrace("生成", `${label(item.source, SOURCE_LABELS)} · 主模型单次入口`);
      addTrace("发送", `${label(item.action_kind, KIND_LABELS)} · ${label(item.action_status || item.status, STATUS_LABELS)}`);
      const facts = document.createElement("dl");
      facts.className = "audit-facts";
      const addFact = (term, value) => {
        const wrapper = document.createElement("div");
        const dt = document.createElement("dt");
        dt.textContent = term;
        const dd = document.createElement("dd");
        dd.textContent = value == null || value === "" ? "未记录" : String(value);
        wrapper.append(dt, dd);
        facts.appendChild(wrapper);
      };
      addFact("触发事件", trigger);
      addFact("触发原文", item.trigger_text);
      addFact("生成入口", label(item.source, SOURCE_LABELS));
      addFact("上下文版本", item.context_version);
      addFact("关系记忆", compactIds(item.relationship_memory_ids));
      addFact("按话题召回记忆", compactIds(item.retrieved_memory_ids));
      addFact("历史条数", item.history_event_count);
      addFact("缓存命中", cacheSummary(item));
      addFact("联网检索", netSummary(item));
      addFact("挂图来源", carriedImagesSummary(contextDetails));
      addFact("引用输入", item.quoted_event_id);
      addFact("引用目标", item.reply_to_event_id);
      const expression = item.expression_intent;
      const expressionSummary = expression
        ? `${expression.kind === "face" ? "QQ 表情" : "消息回应"} · ${expression.key || "—"}${expression.target_event_handle ? ` · 目标 ${expression.target_event_handle}` : ""}`
        : "无";
      addFact("表达动作", expressionSummary);
      addFact("依据等级", item.evidence_level === "full" || item.evidence_level === "complete" ? "结构化完整" : "部分可追溯");
      const note = document.createElement("p");
      note.className = "audit-note";
      note.textContent = item.reason_summary || "仅展示可验证的事件和发送证据，不代表模型隐藏思考过程。";
      article.append(summary, body, profile, recall, trace, facts, note);
      node.appendChild(article);
    });
  }

  function renderTraces() {
    const node = $("trace-events-list");
    node.replaceChildren();
    if (!Array.isArray(state.snapshot && state.snapshot.traces) || !state.snapshot.traces.length) {
      node.textContent = "缺少追踪证据"; node.classList.add("empty-list"); return;
    }
    node.classList.remove("empty-list");
    state.snapshot.traces.forEach((trace) => {
      const entry = document.createElement("details"); entry.className = "audit-entry";
      const summary = document.createElement("summary"); summary.textContent = `${trace.trace_id} · 触发 ${trace.trigger_event_id || "未记录"}`;
      const list = document.createElement("ol"); list.className = "audit-trace";
      (trace.phases || []).forEach((phase) => {
        const item = document.createElement("li");
        item.textContent = `${phase.phase} · ${phase.occurred_at_utc || "—"} · ${JSON.stringify(phase.details || {})}`;
        list.appendChild(item);
      });
      entry.append(summary, list); node.appendChild(entry);
    });
  }

  function renderMemoryRow(item, total) {
    const node = document.createElement("div");
    node.className = "memory-row";
    const header = document.createElement("div");
    header.className = "list-row";
    const key = document.createElement("span");
    key.textContent = label(item.status, STATUS_LABELS);
    const value = document.createElement("strong");
    value.textContent = item.count == null ? "—" : String(item.count);
    header.append(key, value);
    const track = document.createElement("div");
    track.className = "meter";
    const fill = document.createElement("span");
    fill.style.width = `${total ? Math.min(100, (Number(item.count) / total) * 100) : 0}%`;
    fill.className = `meter-fill status-${cssToken(item.status)}`;
    track.appendChild(fill);
    node.append(header, track);
    return node;
  }

  function renderEvents() {
    const body = $("events-body");
    const filter = $("event-filter").value;
    const query = $("event-search").value.trim().toLowerCase();
    const events = state.events.filter((item) => {
      if (filter === "inbound" && item.direction !== "inbound") return false;
      if (filter === "outbound" && item.direction !== "outbound") return false;
      if (filter === "attention" && !["failed", "unknown", "processing", "pending", "dispatched"].includes(item.status)) return false;
      if (!query) return true;
      return [item.text, item.event_id, item.reply_to_event_id, item.kind, item.actor, item.status, item.direction]
        .some((value) => String(value || "").toLowerCase().includes(query));
    });
    body.replaceChildren();
    $("event-count-label").textContent = `${events.length} / ${state.events.length} 条`;
    if (!events.length) {
      const row = document.createElement("tr");
      const cell = document.createElement("td");
      cell.colSpan = 9;
      cell.className = "empty-list";
      cell.textContent = state.events.length ? "没有匹配的事件" : "暂无记录";
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }
    events.forEach((item) => {
      const row = document.createElement("tr");
      row.className = `event-row event-${cssToken(item.status)}`;
      const values = [
        item.sequence,
        item.actor === "mumo" ? "用户" : item.actor === "qichi" ? "角色" : item.actor,
        item.direction === "inbound" ? "入站" : item.direction === "outbound" ? "出站" : item.direction,
        label(item.kind, KIND_LABELS),
        label(item.status, STATUS_LABELS),
        formatTime(item.occurred_at_utc),
        item.text,
        item.reply_to_event_id,
        item.event_id,
      ];
      values.forEach((value, index) => {
        const cell = document.createElement("td");
        if (index === 4) {
          cell.appendChild(statusPill(item.status));
        } else {
          cell.textContent = value == null || value === "" ? "—" : String(value);
        }
        if (index === 6) {
          cell.className = "event-text";
          cell.title = item.text == null ? "" : String(item.text);
        }
        if (index === 7 || index === 8) {
          cell.className = "event-id";
          cell.title = value == null ? "" : String(value);
        }
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  function updateRefreshAge() {
    if (!state.refreshedAt) return;
    $("refresh-age").textContent = `数据 ${relativeTime(state.refreshedAt)}`;
  }

  function recentFailures(snapshot) {
    const traces = Array.isArray(snapshot && snapshot.traces) ? snapshot.traces : [];
    const failures = [];
    traces.forEach((trace) => {
      (trace.phases || []).forEach((phase) => {
        if (phase && (phase.phase === "failure" || phase.phase === "cancelled")) {
          failures.push(phase.occurred_at_utc || null);
        }
      });
    });
    return failures;
  }

  function outboxUnknownCount(snapshot) {
    const statuses = snapshot && snapshot.outbox_status;
    if (!Array.isArray(statuses)) return null;
    let unknown = 0;
    statuses.forEach((item) => {
      if (item && item.status === "unknown") unknown += Number(item.count) || 0;
    });
    return unknown;
  }

  function processStartedAt(snapshot) {
    const marker = (snapshot && snapshot.ready && snapshot.ready.marker) || {};
    return marker.started_at_utc || null;
  }

  function happenedBefore(dateValue, cutoff) {
    if (!dateValue || !cutoff) return false;
    const at = new Date(dateValue).getTime();
    const edge = new Date(cutoff).getTime();
    if (Number.isNaN(at) || Number.isNaN(edge)) return false;
    return at < edge;
  }

  // Concerns are split by one question only: did this happen under the code that
  // is running right now?  A stale state or an old stuck row is history to keep,
  // not something to act on, and mixing the two is what made the panel confusing.
  function collectConcerns(snapshot) {
    const ready = (snapshot && snapshot.ready) || {};
    const health = (snapshot && snapshot.health) || {};
    const cutoff = processStartedAt(snapshot);
    const now = [], history = [];

    if (ready.ok !== true) {
      now.push({ level: "bad", view: "overview", text: "就绪检查未通过", because: ready.reason || "原因未记录" });
    }
    if (health.ok !== true) {
      now.push({ level: "bad", view: "overview", text: "健康检查异常", because: health.reason || "原因未记录" });
    }

    const unknown = outboxUnknownCount(snapshot);
    if (unknown) {
      const rows = (Array.isArray(snapshot.outbox) ? snapshot.outbox : [])
        .filter((item) => item && item.status === "unknown");
      const recent = rows.filter((item) => !happenedBefore(item.updated_at_utc, cutoff));
      const target = recent.length ? now : history;
      target.push({
        level: recent.length ? "warn" : "history",
        view: "overview",
        text: unknown + " 条消息回执未知",
        because: recent.length
          ? "本次运行期间仍有回执未知的发送，需要看一下最近一条"
          : "全部发生在本次启动之前；未知态按设计不会自动重发，也不会被改写成已送达",
        since: rows.length ? rows[rows.length - 1].updated_at_utc : null,
      });
    }

    const failures = recentFailures(snapshot);
    const currentFailures = failures.filter((at) => !happenedBefore(at, cutoff));
    const oldFailures = failures.filter((at) => happenedBefore(at, cutoff));
    if (currentFailures.length) {
      now.push({
        level: "warn", view: "trace", text: "本次运行期间有 " + currentFailures.length + " 次生成或发送失败",
        because: "发生在当前代码运行之后，需要看失败分类", since: currentFailures[currentFailures.length - 1],
      });
    }
    if (oldFailures.length) {
      history.push({
        level: "history", view: "trace", text: oldFailures.length + " 次失败发生在本次启动之前",
        because: "属于已修复的问题（模型标识改名 / 旧发送缺陷），保留只为追溯",
        since: oldFailures[oldFailures.length - 1],
      });
    }

    const initiative = (snapshot && snapshot.initiative) || {};
    const lastResult = initiative.timeline && initiative.timeline.length
      ? initiative.timeline[initiative.timeline.length - 1] : null;
    const features = (snapshot && snapshot.features) || null;
    if (features && features.initiative_enabled === false) {
      history.push({
        level: "history", view: "initiative", text: "主动消息已在配置中关闭",
        because: "这是你关掉的开关，不是故障；开启后会在下一个时段尝试", since: null,
      });
    }
    if (initiative.raw_state === "failed" && !lastResult && !(features && features.initiative_enabled === false)) {
      const stale = happenedBefore(initiative.next_attempt_at_utc, cutoff);
      (stale ? history : now).push({
        level: stale ? "history" : "warn", view: "initiative",
        text: stale ? "主动消息上次记录为失败" : "主动消息当前处于失败状态",
        because: stale
          ? "该记录早于本次启动，且是旧状态的残留；当前运行尚无新的尝试记录"
          : "当前运行中的主动消息失败了，需要看失败原因",
        since: initiative.next_attempt_at_utc || null,
      });
    }
    return { now: now, history: history };
  }

  function renderVerdict(snapshot) {
    const ready = (snapshot && snapshot.ready) || {};
    const health = (snapshot && snapshot.health) || {};
    const concerns = collectConcerns(snapshot);
    const cursor = (snapshot && snapshot.cursor) || {};
    const initiative = (snapshot && snapshot.initiative) || {};
    const band = $("verdict-band");
    band.classList.remove("verdict-ok", "verdict-warn", "verdict-bad");
    let title = "一切正常";
    let detail = "就绪与健康检查通过，没有需要现在处理的问题。";
    let level = "verdict-ok";
    if (ready.ok !== true || health.ok !== true) {
      title = "服务异常";
      detail = ready.ok !== true
        ? "就绪检查未通过：" + (ready.reason || "原因未记录")
        : "健康检查异常：" + (health.reason || "原因未记录");
      level = "verdict-bad";
    } else if (concerns.now.length) {
      title = "有 " + concerns.now.length + " 项需要你处理";
      detail = "服务在线，但下面这些发生在当前运行期间。";
      level = "verdict-warn";
    } else if (concerns.history.length) {
      detail = "没有需要现在处理的问题；下面只有本次启动之前的历史遗留。";
    }
    band.classList.add(level);
    text("verdict-title", title);
    text("verdict-detail", detail + " 最后一条用户消息 " + relativeTime(cursor.last_user_activity_utc)
      + "；下次主动消息 " + relativeTime(initiative.next_attempt_at_utc) + "。");
    text("verdict-issues", concerns.now.length);
    text("concerns-count", concerns.now.length
      ? concerns.now.length + " 项待处理" + (concerns.history.length ? " · " + concerns.history.length + " 项历史" : "")
      : (concerns.history.length ? concerns.history.length + " 项历史遗留" : "无"));
  }

  function renderConcerns(snapshot) {
    const concerns = collectConcerns(snapshot);
    const list = $("concerns-list");
    list.replaceChildren();
    const all = concerns.now.concat(concerns.history);
    $("concerns-card").classList.toggle("all-clear", concerns.now.length === 0);
    if (!all.length) {
      const item = document.createElement("li");
      item.className = "concern concern-ok";
      item.textContent = "没有发现需要处理的问题，也没有历史遗留。";
      list.appendChild(item);
      return;
    }
    all.forEach((concern) => {
      const item = document.createElement("li");
      item.className = "concern concern-" + concern.level;
      const dot = document.createElement("span");
      dot.className = "concern-dot";
      const body = document.createElement("span");
      body.className = "concern-body";
      const headline = document.createElement("strong");
      headline.className = "concern-text";
      headline.textContent = (concern.level === "history" ? "［历史·已修复］" : "［现在］") + concern.text;
      const why = document.createElement("span");
      why.className = "concern-why";
      why.textContent = concern.because + (concern.since ? " · 最近一次 " + formatTime(concern.since) : "");
      body.append(headline, why);
      const jump = document.createElement("button");
      jump.type = "button";
      jump.className = "concern-jump";
      jump.textContent = "查看";
      jump.addEventListener("click", () => setView(concern.view));
      item.append(dot, body, jump);
      list.appendChild(item);
    });
  }

  // 每个开关配一句人话：这些名字本身对使用者没有意义，解释必须跟着数值走。
  const FEATURE_NOTES = {
    "主动消息": "到点她自己来找你说话；可以跳过，夜间静默",
    "Unicode emoji": "😌 这类字符表情",
    "自定义表情包": "收发自定义贴图",
    "QQ 原生表情": "QQ 内置表情（如 /微笑）",
    "消息回应": "给某条消息贴一个 QQ 表情回应，不是发一条聊天消息",
    "记忆自动确认": "新记忆是否自动进入常驻",
    "视觉": "看图能力",
    "外部工具": "联网搜索等外部工具；关闭时她不联网、不调用任何外部服务",
  };

  function featureRows(features) {
    const on = "已启用";
    const off = "已关闭（配置）";
    const flag = (value) => (value === true ? on : value === false ? off : "无证据");
    return [
      { key: "主动消息", value: flag(features.initiative_enabled) },
      { key: "Unicode emoji", value: flag(features.unicode_emoji_enabled) },
      { key: "自定义表情包", value: flag(features.custom_sticker_enabled) },
      { key: "QQ 原生表情", value: flag(features.qq_face_enabled) },
      { key: "消息回应", value: flag(features.message_reaction_enabled) },
      { key: "记忆自动确认", value: features.memory_auto_commit || "无证据" },
      { key: "视觉", value: flag(features.vision_available) },
      { key: "外部工具", value: flag(features.external_tools_available) },
      { key: "文本检索就绪", value: flag(features.external_search_ready) },
      { key: "图搜就绪", value: flag(features.external_image_search_ready) },
    ];
  }

  function renderFeatures(snapshot) {
    const list = $("features-list");
    list.replaceChildren();
    const features = snapshot && snapshot.features;
    text("features-source", features ? "来自进程启动记录" : "无记录");
    text("features-evidence", (snapshot && snapshot.features_evidence) || "无记录（该进程启动时未写入）");
    if (!features) {
      return;
    }
    featureRows(features).forEach((row) => {
      const wrap = document.createElement("div");
      const term = document.createElement("dt");
      term.textContent = row.key;
      const value = document.createElement("dd");
      value.textContent = row.value;
      wrap.append(term, value);
      const note = FEATURE_NOTES[row.key];
      if (note) {
        const hint = document.createElement("small");
        hint.className = "feature-note";
        hint.textContent = note;
        wrap.appendChild(hint);
      }
      list.appendChild(wrap);
    });
  }

  function chatStatus(item) {
    if (item.status === "sent") return { text: "已送达", tone: "ok" };
    if (item.status === "failed") return { text: "发送失败", tone: "bad" };
    if (item.status === "unknown") return { text: "回执未知", tone: "bad" };
    return null;
  }

  function renderChat(snapshot) {
    const timeline = $("chat-timeline");
    timeline.replaceChildren();
    const events = Array.isArray(snapshot && snapshot.recent_events) ? snapshot.recent_events.slice() : [];
    events.sort((a, b) => (Number(a.sequence) || 0) - (Number(b.sequence) || 0));
    text("chat-count", events.length ? "最近 " + events.length + " 条" : "—");
    if (!events.length) {
      const empty = document.createElement("li");
      empty.className = "empty-list";
      empty.textContent = "暂无最近消息";
      timeline.appendChild(empty);
      return;
    }
    events.forEach((item) => {
      const speaker = item.actor === "mumo" ? "用户" : item.actor === "qichi" ? "角色" : (item.actor || "系统");
      const row = document.createElement("li");
      row.className = "chat-row " + (item.actor === "mumo" ? "chat-in" : "chat-out");
      const bubble = document.createElement("div");
      bubble.className = "chat-bubble";
      const delivery = chatStatus(item);
      if (delivery) bubble.classList.add("chat-" + delivery.tone);
      const meta = document.createElement("div");
      meta.className = "chat-meta";
      meta.textContent = speaker + " · " + formatTime(item.occurred_at_utc) + (delivery ? " · " + delivery.text : "");
      const body = document.createElement("p");
      body.className = "chat-text";
      body.textContent = item.text == null || item.text === "" ? "（" + label(item.kind, KIND_LABELS) + "）" : String(item.text);
      bubble.append(meta, body);
      row.appendChild(bubble);
      timeline.appendChild(row);
    });
  }

  function renderSnapshot(snapshot) {
    const ready = snapshot && snapshot.ready || {};
    const runtime = snapshot && snapshot.runtime || {};
    const cursor = snapshot && snapshot.cursor || {};
    const counts = snapshot && snapshot.counts || {};
    const initiative = snapshot && snapshot.initiative || {};
    const outboxStatus = Array.isArray(snapshot.outbox_status) ? snapshot.outbox_status : null;
    const marker = ready.marker && typeof ready.marker === "object" ? ready.marker : {};
    const health = snapshot && snapshot.health || {};
    state.snapshot = snapshot;
    state.events = Array.isArray(snapshot.recent_events) ? snapshot.recent_events : [];
    state.memories = Array.isArray(snapshot.memories) ? snapshot.memories : [];
    state.audits = Array.isArray(snapshot.response_audit) ? snapshot.response_audit : [];
    state.memoryPageInfo = snapshot.memory_page && typeof snapshot.memory_page === "object" ? snapshot.memory_page : null;
    if (state.memoryPageInfo && Number.isInteger(Number(state.memoryPageInfo.page))) state.memoryPage = Number(state.memoryPageInfo.page);
    if (state.memoryPageInfo && Number.isInteger(Number(state.memoryPageInfo.limit))) state.memoryLimit = Number(state.memoryPageInfo.limit);

    const readyOk = ready.ok === true;
    $("ready-badge").textContent = readyOk ? "READY" : "未就绪";
    $("ready-badge").className = `badge ${readyOk ? "ok" : "bad"}`;
    text("ready-reason", ready.reason || (readyOk ? "系统已完成就绪检查" : "就绪检查未通过"));
    text("runtime-model", runtime.model);
    text("runtime-provider", runtime.provider);
    text("runtime-context", formatContext(runtime.context_window));
    text("runtime-owner", runtime.owner_qq);
    text("runtime-bot", runtime.bot_qq);
    text("count-events", counts.events);
    text("event-window", `最近 ${state.events.length} 条`);
    text("count-outbox", outboxStatus === null ? null : counts.outbox);
    text("count-memory", counts.memory);
    text("cursor-context", cursor.context_version);
    text("cursor-sequence", `已处理序号 ${cursor.last_processed_sequence == null ? "—" : cursor.last_processed_sequence}`);
    text("cursor-activity", formatTime(cursor.last_user_activity_utc));
    text("cursor-activity-age", relativeTime(cursor.last_user_activity_utc));
    text("initiative-next", formatTime(initiative.next_attempt_at_utc));
    text("initiative-next-age", relativeTime(initiative.next_attempt_at_utc));
    text("initiative-enabled", initiative.enabled === true ? "已启用" : initiative.enabled === false ? "已关闭" : "—");
    text("initiative-last-activity", formatTime(initiative.last_activity_at_utc));
    text("initiative-state", label(initiative.raw_state, STATUS_LABELS));
    text("initiative-view-enabled", initiative.enabled === true ? "已启用" : initiative.enabled === false ? "已关闭" : "—");
    text("initiative-view-anchor", formatTime(initiative.activity_anchor_utc));
    text("initiative-view-context", initiative.context_version == null ? "未记录" : `v${initiative.context_version}`);
    text("initiative-view-next", formatTime(initiative.next_attempt_at_utc));
    text("initiative-view-activity", formatTime(initiative.last_activity_at_utc));
    text("initiative-view-quiet", initiative.quiet_hours || "未记录");
    text("initiative-view-unanswered", initiative.unanswered == null ? "未记录" : initiative.unanswered);
    text("initiative-view-max-unanswered", initiative.max_unanswered == null ? "未记录" : initiative.max_unanswered);
    text("initiative-view-control", initiative.paused === true ? "已暂停" : initiative.deferred_until_utc ? `已延后至 ${formatTime(initiative.deferred_until_utc)}` : initiative.paused === false ? "未暂停" : "未记录");
    text("initiative-view-state", label(initiative.raw_state, STATUS_LABELS));
    setList("initiative-timeline", initiative.timeline, (item) => {
      const row = document.createElement("div"); row.className = "list-row";
      const name = document.createElement("span"); name.textContent = label(item.category, {due:"到期", claim:"认领", skipped:"跳过", sent:"已送达", failed:"失败", unknown:"未知", cancelled:"已取消"});
      const when = document.createElement("strong"); when.textContent = formatTime(item.at_utc);
      row.append(name, when); return row;
    });
    text("outbox-health", outboxStatus === null ? "统计不可用" : summarizeStatuses(outboxStatus));
    text("memory-health", Array.isArray(snapshot.memory_status) ? summarizeStatuses(snapshot.memory_status) : "暂无记录");
    const outboxItems = Array.isArray(snapshot.outbox) ? snapshot.outbox : [];
    const sentOutbox = outboxStatus === null ? null : countStatuses(outboxStatus, ["sent"]);
    const attentionOutbox = outboxStatus === null ? null : countStatuses(outboxStatus, ["failed", "unknown", "pending", "dispatched"]);
    const totalOutbox = Number(counts.outbox);
    text("outbox-total", outboxStatus === null ? null : Number.isFinite(totalOutbox) ? `${totalOutbox} 条` : null);
    text("outbox-sent-count", sentOutbox);
    text("outbox-attention-count", attentionOutbox);
    text("outbox-detail-label", `最近 ${outboxItems.length} 条，默认收起`);
    text("health-status", health.ok === true ? "正常" : health.reason || "异常");
    $("health-status").className = health.ok === true ? "state-good" : "state-bad";
    $("health-detail").textContent = health.ok === true ? "所有只读数据源可访问" : `数据源异常：${health.reason || "未知原因"}`;

    const component = (id, value) => {
      text(id, value === true ? "正常" : value === false ? "未就绪" : "—");
      $(id).className = value === true ? "state-good" : value === false ? "state-bad" : "";
    };
    component("ws-status", readyOk && Boolean(marker.ws_connection_id));
    component("memory-worker-status", readyOk && Boolean(marker.memory_worker_id));
    component("initiative-worker-status", readyOk && Boolean(marker.initiative_worker_id));

    setList("outbox-list", snapshot.outbox, (item) => {
      const node = document.createElement("div");
      node.className = `outbox-row outbox-${cssToken(item.status)}`;
      const heading = document.createElement("div");
      heading.className = "list-row outbox-heading";
      const action = document.createElement("strong");
      action.textContent = label(item.action_kind, KIND_LABELS);
      heading.append(action, statusPill(item.status));
      node.append(heading, line("尝试次数", item.attempt_count), line("更新时间", formatTime(item.updated_at_utc)));
      return node;
    });
    const memory = Array.isArray(snapshot.memory_status) ? snapshot.memory_status : [];
    const memoryTotal = memory.reduce((sum, item) => sum + (Number(item.count) || 0), 0);
    setList("memory-list", memory, (item) => renderMemoryRow(item, memoryTotal));
    const memoryWorker = snapshot.memory_worker;
    const lastResult = memoryWorker && memoryWorker.last_result;
    if (!lastResult) {
      text("memory-last-result", "最近整理：暂无公开结果");
    } else {
      const details = lastResult.details || {};
      const outcome = details.outcome_kind
        ? `${details.outcome_kind === "no_persistent_memory" ? "无持久记忆" : "有记忆结果"} · ${memoryOutcomeLabel(details.outcome_reason_code)}`
        : (lastResult.failure_category ? `失败 · ${lastResult.failure_category}` : "已完成");
      text("memory-last-result", `最近整理：${outcome} · ${formatTime(lastResult.occurred_at_utc)}`);
    }
    renderMemories();
    renderFragments();
    renderOperations();
    const pageInfo = state.memoryPageInfo;
    text("memory-page-label", pageInfo ? `第 ${pageInfo.page} 页 · ${state.memories.length} / ${pageInfo.total == null ? "—" : pageInfo.total} 条` : "分页信息不可用");
    $("memory-page-prev").disabled = !pageInfo || Number(pageInfo.page) <= 1;
    $("memory-page-next").disabled = !pageInfo || pageInfo.has_more !== true;
    renderResponseAudit();
    renderVerdict(snapshot);
    renderConcerns(snapshot);
    renderChat(snapshot);
    renderFeatures(snapshot);
    renderTraces();
    renderEvents();
    const attention = state.events.filter((item) => ["failed", "unknown", "processing", "pending", "dispatched"].includes(item.status));
    text("ops-audit-count", state.audits.length);
    text("ops-attention-count", attention.length);
    text("ops-latest-event", state.events.length ? formatTime(state.events[0].occurred_at_utc) : "—");
    const qq = state.events.filter((item) => ["poke", "reaction", "face"].includes(item.kind));
    setList("qq-interaction-list", qq, (item) => {
      const row = document.createElement("div");
      row.className = `list-row interaction-${cssToken(item.status)}`;
      const left = document.createElement("span");
      left.textContent = `${label(item.kind, KIND_LABELS)} · ${item.event_id || "—"}`;
      row.append(left, statusPill(item.status));
      return row;
    });
    if (!qq.length) {
      $("qq-interaction-list").textContent = "最近 50 条事件里没有戳一戳 / 表情 / 回应记录";
    }
  }

  function setView(view) {
    state.activeView = view;
    if (window.location.hash.slice(1) !== view) {
      window.history.replaceState(null, "", "#" + view);
    }
    document.querySelectorAll("[data-view]:not(.view-tab)").forEach((node) => {
      node.hidden = node.dataset.view !== view;
    });
    document.querySelectorAll(".view-tab").forEach((button) => {
      const active = button.dataset.view === view;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", active ? "true" : "false");
    });
  }

  async function getJson(url) {
    const requestUrl = url === "/api/snapshot" && state.snapshotQuery ? `${url}?${state.snapshotQuery}` : url;
    const response = await fetch(requestUrl, { method: "GET", headers: { Accept: "application/json" }, cache: "no-store" });
    if (!response.ok) throw new Error(`${url} 返回 HTTP ${response.status}`);
    return response.json();
  }

  async function refresh() {
    if (state.refreshInFlight) return state.refreshInFlight;
    const button = $("refresh-button");
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.textContent = "刷新中…";
    clearError();
    state.refreshInFlight = (async () => {
      try {
        const filter = $("memory-status-filter").value;
        const params = new URLSearchParams({ memory_page: String(state.memoryPage), memory_limit: String(state.memoryLimit) });
        if (filter === "active" || filter === "candidate") params.set("memory_status", filter);
        state.snapshotQuery = params.toString();
        const snapshot = await getJson("/api/snapshot");
        if (!snapshot || typeof snapshot !== "object") throw new Error("快照格式无效");
        renderSnapshot(snapshot);
        state.refreshedAt = Date.now();
        $("last-updated").textContent = `最近刷新：${new Date().toLocaleString("zh-CN", { hour12: false })}`;
        updateRefreshAge();
        $("empty-state").hidden = true;
      } catch (error) {
        showError(
      "面板没能读到运行数据。先看下面这行技术原因；如果是“Failed to fetch”，通常是仪表盘进程没起来。" +
      " 技术原因：" + (error && error.message ? error.message : "未知错误"),
    );
        $("empty-state").hidden = Boolean(state.snapshot);
      } finally {
        state.refreshInFlight = null;
        button.disabled = false;
        button.setAttribute("aria-busy", "false");
        button.textContent = "刷新";
      }
    })();
    return state.refreshInFlight;
  }

  function setAutoRefresh(enabled) {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    try { localStorage.setItem("qichi.dashboard.autoRefresh", enabled ? "1" : "0"); } catch (_) { /* storage is optional */ }
    if (enabled) state.timer = setInterval(refresh, 30000);
  }

  $("refresh-button").addEventListener("click", refresh);
  $("auto-refresh").addEventListener("change", (event) => setAutoRefresh(event.target.checked));
  $("event-filter").addEventListener("change", renderEvents);
  $("event-search").addEventListener("input", renderEvents);
  const resetMemoryPage = () => { state.memoryPage = 1; refresh(); };
  $("memory-status-filter").addEventListener("change", resetMemoryPage);
  $("memory-search").addEventListener("input", () => { state.memoryPage = 1; renderMemories(); });
  $("recall-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("recall-query").value.trim();
    if (text) void runRecallExplanation(text);
  });
  $("memory-page-prev").addEventListener("click", () => { if (state.memoryPage > 1) { state.memoryPage -= 1; refresh(); } });
  $("memory-page-next").addEventListener("click", () => { if (state.memoryPageInfo && state.memoryPageInfo.has_more === true) { state.memoryPage += 1; refresh(); } });
  document.querySelectorAll(".view-tab").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  const deepLinked = window.location.hash.slice(1);
  setView(document.querySelector('.view-tab[data-view="' + deepLinked + '"]') ? deepLinked : state.activeView);
  try {
    $("auto-refresh").checked = localStorage.getItem("qichi.dashboard.autoRefresh") === "1";
  } catch (_) { /* storage is optional */ }
  setAutoRefresh($("auto-refresh").checked);
  state.clockTimer = setInterval(updateRefreshAge, 1000);
  refresh();
})();
