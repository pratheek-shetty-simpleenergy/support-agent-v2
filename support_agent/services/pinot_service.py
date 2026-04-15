from __future__ import annotations

from typing import Any

import requests

from support_agent.config import Settings
from support_agent.runtime.errors import PermanentDependencyError, TransientDependencyError
from support_agent.runtime.retry import run_with_retry


class PinotServiceClient:
    def __init__(self, settings: Settings) -> None:
        self.url = settings.pinot_broker
        self.authorization = settings.pinot_authorization
        self.timeout_seconds = settings.ollama_timeout_seconds
        self.retry_attempts = settings.dependency_retry_attempts
        self.retry_backoff_seconds = settings.dependency_retry_backoff_seconds

    def configured(self) -> bool:
        return bool(self.url and self.authorization)

    def get_telematics_signal_summary(self, vin: str) -> dict[str, Any]:
        sql = f"""
        SELECT vin, event_time, created_at, Vehicle_State, EffectiveSOC, ODO_MeterReading
        FROM CustomerSignals
        WHERE vin = '{_escape_sql_literal(vin)}'
        ORDER BY event_time DESC
        LIMIT 1
        """
        rows = self._execute_sql(sql)
        row = rows[0] if rows else {}
        return {
            "vin": vin,
            "has_signal_data": bool(row),
            "latest_event_time": row.get("event_time"),
            "latest_created_at": row.get("created_at"),
            "vehicle_state": row.get("Vehicle_State"),
            "effective_soc": row.get("EffectiveSOC"),
            "odometer": row.get("ODO_MeterReading"),
        }

    def get_trip_history_summary(self, vin: str, limit: int = 5) -> dict[str, Any]:
        sql = f"""
        SELECT vin, tripId, start_time, end_time, DistanceKM
        FROM Trips
        WHERE vin = '{_escape_sql_literal(vin)}'
        ORDER BY end_time DESC
        LIMIT {int(limit)}
        """
        rows = self._execute_sql(sql)
        latest = rows[0] if rows else {}
        return {
            "vin": vin,
            "has_trip_data": bool(rows),
            "recent_trip_count": len(rows),
            "last_trip_id": latest.get("tripId"),
            "last_trip_end_time": latest.get("end_time"),
            "last_trip_distance_km": latest.get("DistanceKM"),
        }

    def get_charging_history_summary(self, vin: str, limit: int = 5) -> dict[str, Any]:
        sql = f"""
        SELECT vin, startTime, endCharge, initialCharge, totalDuration
        FROM ChargingHistory
        WHERE vin = '{_escape_sql_literal(vin)}'
        ORDER BY startTime DESC
        LIMIT {int(limit)}
        """
        rows = self._execute_sql(sql)
        latest = rows[0] if rows else {}
        return {
            "vin": vin,
            "has_charging_data": bool(rows),
            "recent_charging_session_count": len(rows),
            "last_charging_start_time": latest.get("startTime"),
            "last_charging_end_charge": latest.get("endCharge"),
            "last_charging_duration": latest.get("totalDuration"),
        }

    def healthcheck(self) -> dict[str, Any]:
        return {"configured": self.configured(), "url": self.url}

    def _execute_sql(self, sql: str) -> list[dict[str, Any]]:
        if not self.configured():
            raise PermanentDependencyError("Pinot service is not configured.")

        def do_request() -> list[dict[str, Any]]:
            try:
                response = requests.post(
                    self.url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": self.authorization or "",
                    },
                    json={"sql": sql.strip()},
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise TransientDependencyError(f"Pinot request failed: {exc}") from exc

            if response.status_code >= 500:
                raise TransientDependencyError(f"Pinot returned {response.status_code}: {response.text}")
            if response.status_code >= 400:
                raise PermanentDependencyError(f"Pinot returned {response.status_code}: {response.text}")

            payload = response.json()
            exceptions = payload.get("exceptions") or []
            if exceptions:
                raise PermanentDependencyError(f"Pinot query errors: {exceptions}")
            result_table = payload.get("resultTable") or {}
            data_schema = result_table.get("dataSchema") or {}
            column_names = data_schema.get("columnNames") or []
            rows = result_table.get("rows") or []
            if not isinstance(column_names, list) or not isinstance(rows, list):
                return []
            return [_map_row(column_names, row) for row in rows if isinstance(row, list)]

        return run_with_retry(
            do_request,
            attempts=self.retry_attempts,
            backoff_seconds=self.retry_backoff_seconds,
        )


def _map_row(column_names: list[str], row: list[Any]) -> dict[str, Any]:
    return {
        str(column_names[index]): row[index]
        for index in range(min(len(column_names), len(row)))
    }


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")
