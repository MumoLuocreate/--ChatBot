"""卡C：指向检索——两字窗口只放宽到 episode，且排除时间指示词。

判据见 doc/方案-20260922-指向检索.md。三条硬要求：
  ①他指向一件**发生过的事**时，检索要能翻到（修复前整句指向候选 0 条）；
  ②日常话语里的偶然重合不得把无关记忆（尤其 preference）拉进来——P116 红线；
  ③时间指示词（今天/明天/下午…）不是"指向"，不许用来放宽——它们指时间，
    而指向一件事用的是命名实体（团建/校区/文档/TTS）。
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qichi.memory.dates import names_a_time_term  # noqa: E402
from qichi.memory.retriever import MemoryRetriever  # noqa: E402
from test_memory_repository import NOW, event, memory, repositories  # noqa: E402

QUESTION = "我前几天跟你说的团建那事，后来怎么样了"


def _seed_episode(repositories):
    _, events, memories = repositories
    # 原话与问句**只共享两字关键词「团建」**、不共享任何三字窗口——生产就是这个形状。
    source = events.insert(event(
        "e-zhaoxin", "新生报到那天我在团建处守着，从早八点待到晚六点",
        conversation_id="conversation-a"))
    memories.create(memory(
        "m-zhaoxin", source, status="active", type="episode", temporal_scope="historical",
        fact="2026年9月12日，用户因学校新生报到，在战队团建处做宣传，从早八点待到晚六点。"))
    return MemoryRetriever(database=repositories[0], candidate_limit=24, context_limit=12)


def test_pointing_sentence_reaches_the_episode(repositories):
    """命中：整句指向必须能翻到（修复前这里是 0 条候选）。"""

    retriever = _seed_episode(repositories)
    result = retriever.retrieve("conversation-a", QUESTION, NOW)
    assert "m-zhaoxin" in {record.memory_id for record in result.candidates}


def test_two_character_window_is_what_makes_it_reachable(repositories):
    """成立的原因是「两字窗口喂对了路」，不是运气。"""

    retriever = _seed_episode(repositories)
    windows = MemoryRetriever._query_bigrams((QUESTION,))
    assert "团建" in windows
    assert "m-zhaoxin" in retriever._like_candidate_ids(
        "conversation-a", windows, NOW, record_type="episode"
    )


def test_a_preference_is_never_widened_by_two_character_windows(repositories):
    """不误判（P116 红线）：两字窗口不许把偏好拉进来。"""

    _, events, memories = repositories
    source = events.insert(event("e-soft", "我觉得可以考虑换一个说法，偶尔这样一次也行"))
    memories.create(memory("soft-tone", source, fact="用户偶尔希望角色换个说法一次"))
    retriever = MemoryRetriever(database=repositories[0])
    result = retriever.retrieve(
        "conversation-a", "我今天空腹喝的苦涩…我考虑换一种说法", NOW)
    assert "soft-tone" not in {record.memory_id for record in result.candidates}


def test_a_profile_time_word_does_not_widen_to_an_episode(repositories):
    """不误判：时间指示词不许放宽（按语言类别，不是词频）。

    实测「明天下午两点提醒我开会」曾把 3 条退场经历拉进提示，泄漏词是「明天」「下午」；
    而「明天」的 episode 出现次数与三个真目标完全相同（都是 2），词频分不开它们。
    """

    _, events, memories = repositories
    source = events.insert(event(
        "e-tomorrow", "我明天还要带两拨新生参观", conversation_id="conversation-a"))
    memories.create(memory(
        "m-tomorrow", source, status="active", type="episode", temporal_scope="historical",
        fact="2026年9月18日，用户明天周六还要带两拨新生参观。"))
    retriever = MemoryRetriever(database=repositories[0])

    assert names_a_time_term("明天") and names_a_time_term("下午")
    assert not any(names_a_time_term(word) for word in ("团建", "校区", "文档", "TTS"))

    result = retriever.retrieve("conversation-a", "明天下午两点提醒我开会", NOW)
    assert "m-tomorrow" not in {r.memory_id for r in result.candidates}


def test_casual_messages_do_not_reach_the_episode(repositories):
    """不误判：日常闲聊不得把这条无关经历拉进来。"""

    retriever = _seed_episode(repositories)
    for casual in ("在吗", "今天有点累", "好无聊啊"):
        result = retriever.retrieve("conversation-a", casual, NOW)
        assert "m-zhaoxin" not in {record.memory_id for record in result.candidates}, casual


def test_a_generic_window_is_not_allowed_to_widen(repositories):
    """泛词不许放宽：在太多 episode 里出现过的两字词没有指向性（上限 5）。"""

    database, events, memories = repositories
    for index in range(7):  # 超过 _MAX_WINDOW_DOCUMENT_FREQUENCY
        source = events.insert(event(
            f"e-generic-{index}", f"聊了第 {index} 件事", conversation_id="conversation-a"))
        memories.create(memory(
            f"m-generic-{index}", source, status="active", type="episode",
            temporal_scope="historical", fact=f"2026年9月{index + 1}日，聊到第 {index} 件事。"))
    retriever = MemoryRetriever(database=repositories[0])
    assert "聊了" in retriever._frequent_windows(("聊了",), NOW)


def test_ascii_keywords_become_windows():
    """TTS / AI 这类纯字母关键词也必须是窗口（第一版只取汉字对，TTS 那格结构上漏掉）。"""

    windows = MemoryRetriever._query_bigrams(("上次那个 TTS 的事情怎么样了",))
    assert "TTS" in windows


def test_bigrams_take_chinese_pairs_and_ascii_runs():
    """单元：汉字按对取，纯字母按连续串取。"""

    fragments = MemoryRetriever._query_bigrams(("团建那事ABC，de",))
    assert set(fragments) == {"团建", "建那", "那事", "ABC", "de"}
