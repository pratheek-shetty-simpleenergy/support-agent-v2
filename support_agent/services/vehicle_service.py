from __future__ import annotations

from typing import Any

import requests

from support_agent.config import Settings
from support_agent.runtime.errors import PermanentDependencyError, TransientDependencyError
from support_agent.runtime.retry import run_with_retry


GQL_GET_LASTSEEN = """
query GetLastseenOfVehicle($vin: String!) {
  getLastseenOfVehicle(vin: $vin) {
    data {
      vin
      last_seen_epoch
      last_seen
      isActive
    }
    success
  }
}
"""


class VehicleServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.url = settings.vehicle_service_url
        self.api_key = settings.vehicle_service_apikey
        self.timeout_seconds = settings.ollama_timeout_seconds
        self.retry_attempts = settings.dependency_retry_attempts
        self.retry_backoff_seconds = settings.dependency_retry_backoff_seconds

    def configured(self) -> bool:
        return bool(self.url and self.api_key)

    def get_vehicle_last_seen(self, vin: str) -> dict[str, Any] | None:
        if not self.configured():
            raise PermanentDependencyError("Vehicle service is not configured.")

        def do_request() -> dict[str, Any] | None:
            try:
                response = requests.post(
                    self.url,
                    headers={
                        "Content-Type": "application/json",
                        "x-api-key": self.api_key or "",
                    },
                    json={"query": GQL_GET_LASTSEEN, "variables": {"vin": vin}},
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise TransientDependencyError(f"Vehicle service request failed: {exc}") from exc

            if response.status_code >= 500:
                raise TransientDependencyError(f"Vehicle service returned {response.status_code}: {response.text}")
            if response.status_code >= 400:
                raise PermanentDependencyError(f"Vehicle service returned {response.status_code}: {response.text}")

            payload = response.json()
            if payload.get("errors"):
                raise PermanentDependencyError(f"Vehicle service GraphQL errors: {payload['errors']}")

            result = ((payload.get("data") or {}).get("getLastseenOfVehicle") or {})
            if not result.get("success"):
                return None
            data = result.get("data") or {}
            if not isinstance(data, dict):
                return None
            return {
                "vin": data.get("vin"),
                "last_seen_epoch": data.get("last_seen_epoch"),
                "last_seen": data.get("last_seen"),
                "is_active": data.get("isActive"),
            }

        return run_with_retry(
            do_request,
            attempts=self.retry_attempts,
            backoff_seconds=self.retry_backoff_seconds,
        )

    def healthcheck(self) -> dict[str, Any]:
        return {"configured": self.configured(), "url": self.url}
