from __future__ import annotations

from support_agent.config import Settings
from support_agent.services.pinot_service import PinotServiceClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_pinot_service_get_telematics_signal_summary(monkeypatch) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers or {}, json or {}))
        return FakeResponse(
            200,
            {
                "resultTable": {
                    "dataSchema": {
                        "columnNames": ["vin", "event_time", "created_at", "Vehicle_State", "EffectiveSOC", "ODO_MeterReading"],
                    },
                    "rows": [["VIN-1", 1710000000000, 1710000000500, 3, 72.5, 1842.0]],
                }
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    client = PinotServiceClient(
        Settings(
            PINOT_BROKER="https://pinot-broker-stage.simpleenergy.in/query",
            PINOT_AUTHORIZATION="Basic test",
        )
    )
    payload = client.get_telematics_signal_summary("VIN-1")
    assert payload == {
        "vin": "VIN-1",
        "has_signal_data": True,
        "latest_event_time": 1710000000000,
        "latest_created_at": 1710000000500,
        "vehicle_state": 3,
        "effective_soc": 72.5,
        "odometer": 1842.0,
    }
    assert calls[0][0] == "https://pinot-broker-stage.simpleenergy.in/query"
    assert calls[0][1]["Authorization"] == "Basic test"
    assert "CustomerSignals" in calls[0][2]["sql"]
