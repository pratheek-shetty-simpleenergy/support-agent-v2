from typing import Any


def format_match_context(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "No relevant reference context was retrieved."

    chunks: list[str] = []
    for index, match in enumerate(matches, start=1):
        metadata = match.get("metadata", {})
        source = metadata.get("source", "unknown")
        title = metadata.get("title", "untitled")
        text = metadata.get("text", "")
        score = match.get("score", 0.0)
        chunks.append(f"[{index}] source={source} title={title} score={score:.3f}\n{text}".strip())
    return "\n\n".join(chunks)
