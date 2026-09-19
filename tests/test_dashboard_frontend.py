import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"


def test_dashboard_frontend_files_exist():
    assert all((DASHBOARD / name).is_file() for name in ("index.html", "app.js", "styles.css"))


def test_dashboard_is_local_read_only_and_has_required_hooks():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "https://" not in html + js
    assert "POST" not in js and "DELETE" not in js and "PUT" not in js
    assert 'getJson("/api/snapshot"' in js
    for hook in ("refresh-button", "auto-refresh", "ready-badge", "runtime-model", "ws-status", "memory-worker-status", "initiative-worker-status", "events-body", "event-filter", "event-search", "refresh-age", "outbox-list", "outbox-details", "outbox-detail-label", "memory-list", "initiative-next", "initiative-view-activity", "memories-list", "memory-status-filter", "memory-search", "response-audit-list"):
        assert f'id="{hook}"' in html
    assert "audit-trace" in js and "隐藏思考" in html
    for view, label in (("overview", "总览"), ("trace", "对话"), ("memory", "记忆"), ("presence", "在场"), ("ops", "运维")):
        assert f'data-view="{view}"' in html
        assert label in html
    assert "function setView" in js and "view-tab" in js
    assert 'document.querySelectorAll("[data-view]:not(.view-tab)")' in js


def test_the_retired_tabs_are_merged_into_presence():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert 'data-view="initiative"' not in html, "主动消息 与 QQ 互动 合并为「在场」"
    assert 'data-view="qq"' not in html
    assert html.count('data-view="presence"') >= 2, "在场页承载两块证据"


def test_every_view_tab_has_a_section_of_its_own():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    tabs = re.findall(r'class="view-tab[^"]*" data-view="([a-z]+)"', html)
    seen = re.findall(r'data-view="([a-z]+)"', html)
    assert tabs == ["overview", "trace", "memory", "presence", "ops"]
    for view in tabs:
        assert seen.count(view) >= 2, f"{view} 必须至少有一个自己的区块，不能点进去是空页"


def test_the_palette_is_declared_once_as_design_tokens():
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    root = re.search(r":root\s*\{(.*?)\}", css, re.DOTALL)
    assert root is not None, "配色必须以令牌形式集中声明"
    block = root.group(1)
    for token in ("--bg:", "--surface-1:", "--surface-2:", "--border:", "--text-1:", "--text-2:",
                  "--accent:", "--ok:", "--warn:", "--error:", "--unknown:",
                  "--radius-sm:", "--radius-lg:", "--space-2:", "--space-4:"):
        assert token in block, f"缺少设计令牌 {token}"


def test_component_statuses_are_derived_from_ready_evidence_ids():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    for evidence_id in ("ws_connection_id", "memory_worker_id", "initiative_worker_id"):
        assert f"marker.{evidence_id}" in js
    for invented_flag in ("marker.ws_ready", "marker.memory_worker_ready", "marker.initiative_scheduler_ready"):
        assert invented_flag not in js


def test_refresh_uses_snapshot_health_without_a_duplicate_health_snapshot():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert 'getJson("/api/snapshot"' in js
    assert 'getJson("/api/health")' not in js
    assert "snapshot.health = health" not in js
    assert 'const health = snapshot && snapshot.health || {};' in js
    assert 'throw new Error("快照格式无效")' in js
    assert "面板没能读到运行数据" in js and "Failed to fetch" in js, "错误横幅要说人话，同时保留技术原因"


def test_every_literal_dom_hook_used_by_javascript_exists_in_html():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    declared_ids = set(re.findall(r'\bid="([^"]+)"', html))
    referenced_ids = set(re.findall(r'\$\("([^"]+)"\)', js))
    assert referenced_ids <= declared_ids


def test_dashboard_does_not_render_secrets_and_has_responsive_css():
    source = "\n".join((DASHBOARD / name).read_text(encoding="utf-8") for name in ("index.html", "app.js", "styles.css"))
    assert "api_key" not in source.lower()
    assert "access_token" not in source.lower()
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert "@media" in css and "max-width" in css
    assert "[hidden] { display: none !important; }" in css


def test_trace_phase_payloads_wrap_without_expanding_the_page():
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    trace_item_rule = re.search(r"\.audit-trace li\s*\{([^}]*)\}", css, re.DOTALL)
    assert trace_item_rule is not None
    assert "overflow-wrap: anywhere" in trace_item_rule.group(1)
    trace_summary_rule = re.search(r"#trace-events-list summary\s*\{([^}]*)\}", css, re.DOTALL)
    assert trace_summary_rule is not None
    assert "overflow-wrap: anywhere" in trace_summary_rule.group(1)
    table_wrap_rule = re.search(r"\.table-wrap\s*\{([^}]*)\}", css, re.DOTALL)
    assert table_wrap_rule is not None
    assert "max-width: 100%" in table_wrap_rule.group(1)
    assert "overflow: auto" in table_wrap_rule.group(1)


def test_narrow_shell_full_bleed_sections_match_the_12px_breakpoint_padding():
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    narrow = re.search(r"@media \(max-width: 420px\) \{(.*?)\n\}", css, re.DOTALL)
    assert narrow is not None
    block = narrow.group(1)
    assert ".view-tabs { margin-right: -12px; padding-right: 12px; }" in block
    assert ".status-band { margin-left: -12px; margin-right: -12px; padding-left: 12px; padding-right: 12px; }" in block


def test_memory_uses_real_assessment_fields_and_public_reason_codes():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    source = html + js
    for field in ("certainty", "importance", "temporal_scope", "recall_scope", "assessment_reason_code", "supersedes", "assessed_at"):
        assert field in source
    assert "原因码说明" in source
    assert "证据时间线" in source
    assert "待确认" in html and "memory-status-filter" in html
    for legacy in ("纠正 = 最高", "偏好 = 中", "常驻关系状态", "按话题检索", "不代表情感重要度"):
        assert legacy not in source


def test_the_conversation_page_shows_what_she_actually_saw():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert "renderContextProfile" in js and "contextPhaseDetails" in js
    assert "category_tokens" in js, "剖面必须来自真实观测的 token 分类"
    for category in ("memory_index", "memory_details"):
        assert category in js, f"{category} 必须在剖面上有名字"
    assert "按你点名的日期定位" in js and "已标注为推定" in js
    assert ".ctx-bar" in css and ".ctx-chip-guess" in css
    assert "article.append(summary, body, profile, recall, trace, facts, note)" in js, (
        "剖面必须按装配顺序入树；insertBefore 一个尚未 append 的节点会在运行时抛错"
    )


def test_the_view_tab_is_shareable_and_survives_a_refresh():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "window.location.hash" in js and "replaceState" in js


def test_no_light_surfaces_survive_outside_the_token_layer():
    """暗色面板里任何一行亮色字面量都会变成一个刺眼的块（用户已两次撞上）。"""
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    lines = css.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip().startswith(":root"))
    end = next(index for index in range(start, len(lines)) if lines[index].strip() == "}")

    def luminance(red, green, blue):
        def channel(value):
            value /= 255
            return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
        return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)

    offenders = []
    for index, line in enumerate(lines):
        if start <= index <= end:
            continue
        for match in re.finditer(r"#[0-9a-fA-F]{6}\b", line):
            digits = match.group(0)[1:]
            red, green, blue = (int(digits[offset:offset + 2], 16) for offset in (0, 2, 4))
            if luminance(red, green, blue) > 0.25:
                offenders.append(f"行 {index + 1}: {match.group(0)}")
    assert offenders == [], "这些亮色字面量必须换成令牌：" + "；".join(offenders)


def test_the_memory_page_reads_episodes_and_explains_recall():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    for hook in ("fragment-timeline", "fragments-count", "recall-form", "recall-query", "recall-result"):
        assert f'id="{hook}"' in html, f"缺少 {hook}"
    assert "renderFragments" in js and 'data-view="memory"' in html
    assert "/api/fragment?id=" in js and "/api/recall?q=" in js, "原文与复算都必须按需只读拉取"
    assert "成人内容，默认折叠" in js and "fragment-reveal" in js, "成人原文必须默认折叠"
    assert "词表命中" in js and "词表未命中" in js, "复算必须给出明确判据"
    assert ".fragment-card" in css and ".recall-result" in css


def test_the_memory_page_never_writes_and_never_invents_a_verdict():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "POST" not in js and "PUT" not in js and "DELETE" not in js
    assert 'getJson("/api/fragment?id=" + encodeURIComponent(item.fragment_id)' in js


def test_the_ops_page_reports_build_database_and_watermarks():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    for hook in ("ops-build-status", "ops-build-code", "ops-build-marker", "ops-build-note",
                 "ops-db-name", "ops-db-schema", "ops-db-size", "ops-db-wal",
                 "ops-backups", "ops-watermarks"):
        assert f'id="{hook}"' in html, f"缺少 {hook}"
    assert "renderOperations" in js and "operations.build" in js
    assert "需要重启" in js, "指纹不一致必须说人话，而不是让用户猜"
    assert "formatBytes" in js and ".ops-block" in css


def test_the_conversation_page_shows_what_she_remembered_and_why():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert "renderMemoryRecall" in js and "她这一轮想起了什么" in js
    assert "relationship_refs" in js and "retrieved_refs" in js, "常驻与按话题想起必须分别展示内容"
    assert "describeMemoryReason" in js and "相关度" in js, "必须能看出为什么想起这一条"
    assert "三字重合" in js, "命中理由必须说人话，不能直接抛 trigram_overlap 这种内部串"
    assert "读这一段的原话" in js and "/api/fragment?id=" in js, "明细原文要能在这一轮里直接读"
    assert "这一轮没有把原话摊开给她" in js, "没展开时必须说清原因，而不是留一个空"


def test_panel_explains_its_own_vocabulary():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    for term in ("片段索引", "明细原文", "未展开明细原文", "关系记忆（常驻）", "按话题想起", "消息回应", "外部工具", "直接引用"):
        assert term in html, f"术语表缺少 {term}"
    assert "不是发一条聊天消息" in html, "消息回应必须与聊天消息区分开"
    assert "她不联网" in html and "她不联网" in js, "外部工具要说清关闭意味着什么"


def test_memory_filter_does_not_mislabel_retrieved_memories_as_candidates():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "retrieved_memory_ids" in js
    assert "candidate_memory_ids" not in js
    assert "候选记忆" not in js


def test_trace_is_single_four_phase_view_and_technical_details_are_collapsed():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "触发" in html and "上下文" in html and "生成" in html and "发送" in html
    assert "原始技术详情" in html
    assert "<details" in html
    assert "prompt" not in (html + js).lower()
    assert "completion" not in (html + js).lower()
    assert "隐藏思考" in html


def test_memory_pagination_uses_snapshot_page_metadata_and_resets_on_filters():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "memory_page" in js and "memory_limit" in js and "memory_status" in js
    assert "snapshot.memory_page" in js
    assert all(hook in html for hook in ("memory-page-prev", "memory-page-next", "memory-page-label"))
    assert "has_more" in js and "memoryPage" in js
    assert "全部记忆" not in js
    assert "state.memoryPage = 1" in js


def test_memory_detail_exposes_audit_and_confirmation_public_fields_only():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    source = html + js
    assert "audit_events" in source and "confirmation_presentations" in source
    assert "审核事件" in source and "确认展示" in source
    assert "prompt" not in source.lower() and "reasoning" not in source.lower()
    assert "JSON.stringify(item" not in js


def test_memory_detail_uses_backend_audit_and_presentation_field_names():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert '"audit_event_id", "action", "before_status", "after_status", "assessment_reason_code", "occurred_at_utc"' in js
    assert '"presentation_id", "fragment_key", "trigger_event_id", "context_version", "presented_at_utc"' in js
    for legacy in ("expires_at_utc", '"event_id", "action", "actor", "occurred_at_utc", "status"'):
        assert legacy not in js


def test_outbox_overview_uses_full_status_aggregate_and_keeps_recent_details_separate():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    assert "snapshot.outbox_status" in js
    assert 'text("outbox-total"' in js and "snapshot.counts" in js
    assert "summarizeStatuses(outboxStatus)" in js
    assert "countStatuses(outboxStatus" in js
    assert '"统计不可用"' in js
    assert 'text("count-outbox", outboxStatus === null ? null' in js
    assert 'text("outbox-total", outboxStatus === null ? null' in js
    assert 'counts.has("pending")' in js and 'counts.has("dispatched")' in js
    assert "outboxItems.filter" not in js
    assert 'setList("outbox-list", snapshot.outbox' in js
    assert "最近 ${outboxItems.length} 条，默认收起" in js


def test_overview_leads_with_one_verdict_and_a_concerns_list():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    for hook in ("verdict-band", "verdict-title", "verdict-detail", "verdict-issues",
                 "concerns-card", "concerns-list", "chat-card", "chat-timeline"):
        assert f'id="{hook}"' in html, hook
    for function in ("function renderVerdict", "function renderConcerns", "function renderChat"):
        assert function in js, function
    for call in ("renderVerdict(snapshot)", "renderConcerns(snapshot)", "renderChat(snapshot)"):
        assert call in js, call


def test_verdict_and_concerns_are_derived_from_real_evidence_only():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    # The verdict must come from readiness and health, never from an invented flag.
    assert "snapshot.ready" in js and "snapshot.health" in js
    assert "一切正常" in js and "需要你处理" in js and "服务异常" in js
    # Concerns are computed from the same public aggregates the rest of the panel uses.
    assert "outboxStatus" in js and "snapshot.initiative" in js and "snapshot.traces" in js
    assert "unknown" in js
    for invented in ("assume_healthy", "fake_status", "marker.ok"):
        assert invented not in js


def test_chat_timeline_renders_speakers_time_and_delivery_state():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    assert "chat-bubble" in js and "chat-bubble" in css
    assert "用户" in js and "角色" in js
    for state in ("已送达", "发送失败", "回执未知"):
        assert state in js, state
    assert ".chat-row" in css and ".chat-meta" in css


def test_new_cards_stay_responsive_and_bounded():
    css = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    verdict_rule = re.search(r"\.verdict-band\s*\{([^}]*)\}", css, re.DOTALL)
    assert verdict_rule is not None
    chat_text_rule = re.search(r"\.chat-text\s*\{([^}]*)\}", css, re.DOTALL)
    assert chat_text_rule is not None
    assert "overflow-wrap: anywhere" in chat_text_rule.group(1)
    assert ".chat-timeline" in css


def test_concerns_separate_current_problems_from_history():
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    # The split must key off evidence that the running code started, never off a guess.
    assert "marker.started_at_utc" in js
    assert "function processStartedAt" in js and "function happenedBefore" in js
    assert "concerns.now" in js and "concerns.history" in js
    for label in ("［现在］", "［历史·已修复］", "需要你处理", "历史遗留"):
        assert label in js, label
    assert "全部发生在本次启动之前" in js
    assert "concern-why" in js


def test_configuration_switches_come_from_the_running_process():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    js = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    for hook in ("features-card", "features-list", "features-evidence", "features-source"):
        assert f'id="{hook}"' in html, hook
    assert "function renderFeatures" in js and "renderFeatures(snapshot)" in js
    assert "配置开关" in html
    assert "已关闭（配置）" in js
    assert "snapshot.features" in js and "features_evidence" in js
    # A switch the owner turned off must be reported as history, never as a fault.
    assert "主动消息已在配置中关闭" in js

