from __future__ import annotations

from support_agent.config import Settings
from support_agent.services.vehicle_service import VehicleServiceClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_vehicle_service_get_last_seen(monkeypatch) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers or {}, json or {}))
        return FakeResponse(
            200,
            {
                "data": {
                    "getLastseenOfVehicle": {
                        "success": True,
                        "data": {
                            "vin": "VIN-1",
                            "last_seen_epoch": 1710000000,
                            "last_seen": "2026-04-15T10:00:00Z",
                            "isActive": True,
                        },
                    }
                }
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    client = VehicleServiceClient(
        Settings(
            VEHICLE_SERVICE_APIKEY="test-key",
            VEHICLE_SERVICE_URL="https://vehicle-stage.simpleenergy.in/api/graphql",
        )
    )
    payload = client.get_vehicle_last_seen("VIN-1")
    assert payload == {
        "vin": "VIN-1",
        "last_seen_epoch": 1710000000,
        "last_seen": "2026-04-15T10:00:00Z",
        "is_active": True,
    }
    assert calls[0][0] == "https://vehicle-stage.simpleenergy.in/api/graphql"
    assert calls[0][1]["x-api-key"] == "test-key"
