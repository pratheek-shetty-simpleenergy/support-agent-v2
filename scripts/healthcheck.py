from __future__ import annotations

import json

from support_agent.runtime import configure_logging
from support_agent.services.healthcheck import run_healthcheck


def main() -> None:
    configure_logging()
    print(json.dumps(run_healthcheck(), indent=2, default=str))


if __name__ == "__main__":
    main()
