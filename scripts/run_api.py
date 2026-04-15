from __future__ import annotations

from support_agent.api.app import app


def main() -> None:
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise RuntimeError("uvicorn is not installed. Install project dependencies before running the API.") from exc

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
