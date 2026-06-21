import os

from tavily import TavilyClient

try:
    from config import TAVILY_API_KEY
except ImportError:
    TAVILY_API_KEY = None

_client = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        key = TAVILY_API_KEY or os.environ.get("TAVILY_API_KEY")
        if not key:
            raise RuntimeError(
                "TAVILY_API_KEY가 설정되지 않았습니다. config.py 또는 환경변수를 확인하세요."
            )
        _client = TavilyClient(api_key=key)
    return _client


# Live API 의 FunctionResponse 페이로드가 너무 크면 서버가 1008(policy violation)
# 으로 세션을 끊는다. 모델이 음성으로 요약해서 안내해야 하므로 길게 줄 필요도
# 없다. 항목 본문은 짧게 자르고 전체 결과도 상한을 둔다.
_PER_ITEM_CONTENT_CHARS = 240
_MAX_TOTAL_CHARS = 2000


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip().replace("\r", " ").replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def search_web(query: str, max_results: int = 5) -> str:
    """Tavily로 웹 검색 후 모델이 읽기 좋게 정리된 문자열을 반환한다."""
    max_results = max(1, min(max_results or 5, 5))
    try:
        resp = _get_client().search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )
    except Exception as error:
        return f"검색 실패: {error}"

    lines: list[str] = []
    answer = (resp.get("answer") or "").strip()
    if answer:
        lines.append(f"요약: {_truncate(answer, 600)}")
    for idx, item in enumerate(resp.get("results", [])[:max_results], start=1):
        title = _truncate(item.get("title") or "", 120)
        content = _truncate(item.get("content") or "", _PER_ITEM_CONTENT_CHARS)
        url = (item.get("url") or "").strip()
        lines.append(f"{idx}. {title}\n   {content}\n   출처: {url}")

    if not lines:
        return "검색 결과가 없습니다."

    out = "\n".join(lines)
    if len(out) > _MAX_TOTAL_CHARS:
        out = out[: _MAX_TOTAL_CHARS - 1].rstrip() + "…"
    return out
