"""联网（外部检索）：只在 net.enabled 打开后才参与对话装配。"""

from qichi.net.tools import (
    IMAGE_SEARCH_TOOL,
    WEB_SEARCH_TOOL,
    SearchToolRunner,
    ToolRunResult,
    tool_plan,
)
from qichi.net.image_search import (
    DEFAULT_SEARCH_TYPE,
    SEARCH_TYPES,
    ImageMatch,
    ImageSearchOutcome,
    SerpApiLensClient,
    WebImageSearchError,
    parse_matches,
    render_image_block,
)
from qichi.net.search import (
    MAX_QUERY_CHARS,
    SearchOutcome,
    SearchResult,
    TavilySearchClient,
    WebSearchError,
    clean_query,
    parse_results,
    render_external_block,
)

__all__ = [
    "DEFAULT_SEARCH_TYPE",
    "IMAGE_SEARCH_TOOL",
    "SearchToolRunner",
    "ToolRunResult",
    "tool_plan",
    "WEB_SEARCH_TOOL",
    "ImageMatch",
    "ImageSearchOutcome",
    "MAX_QUERY_CHARS",
    "SEARCH_TYPES",
    "SerpApiLensClient",
    "WebImageSearchError",
    "parse_matches",
    "render_image_block",
    "SearchOutcome",
    "SearchResult",
    "TavilySearchClient",
    "WebSearchError",
    "clean_query",
    "parse_results",
    "render_external_block",
]
