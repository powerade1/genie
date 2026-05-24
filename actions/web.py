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


def search_web(query: str, max_results: int = 5) -> str:
    """Tavily로 웹 검색 후 모델이 읽기 좋게 정리된 문자열을 반환한다."""
    try:
        resp = _get_client().search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=True,
        )
    except Exception as error:
        return f"검색 실패: {error}"

    lines = []
    if resp.get("answer"):
        lines.append(f"요약: {resp['answer']}")
    for idx, item in enumerate(resp.get("results", [])[:max_results], start=1):
        title = item.get("title", "")
        content = (item.get("content") or "").strip()
        url = item.get("url", "")
        lines.append(f"{idx}. {title}\n   {content}\n   출처: {url}")

    return "\n".join(lines) if lines else "검색 결과가 없습니다."
