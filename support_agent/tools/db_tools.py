from __future__ import annotations

from support_agent.db.repositories import BusinessDbRepository
from support_agent.schemas.tool import ToolResult
from support_agent.tools.base import ToolRegistry


def build_business_db_tools(repository: BusinessDbRepository) -> ToolRegistry:
    registry = ToolRegistry()

    def get_user_profile(user_id: str) -> ToolResult:
        payload = repository.get_user_profile(user_id=user_id)
        return ToolResult(name="get_user_profile", success=payload is not None, payload={"user_profile": payload or {}})

    def get_user_profile_by_mobile(mobile: str) -> ToolResult:
        payload = repository.get_user_profile_by_mobile(mobile=mobile)
        return ToolResult(name="get_user_profile_by_mobile", success=payload is not None, payload={"user_profile": payload or {}})

    def get_booking_details(booking_id: str) -> ToolResult:
        payload = repository.get_booking_details(booking_id=booking_id)
        return ToolResult(name="get_booking_details", success=payload is not None, payload={"booking_details": payload or {}})

    def get_order_details(order_id: str | None = None, order_number: str | None = None) -> ToolResult:
        payload = repository.get_order_details(order_id=order_id, order_number=order_number)
        return ToolResult(name="get_order_details", success=payload is not None, payload={"order_details": payload or {}})

    def get_payment_status(payment_id: str) -> ToolResult:
        payload = repository.get_payment_status(payment_id=payment_id)
        return ToolResult(name="get_payment_status", success=payload is not None, payload={"payment_status": payload or {}})

    def get_order_payment_status(order_id: str) -> ToolResult:
        payload = repository.get_order_payment_status(order_id=order_id)
        return ToolResult(name="get_order_payment_status", success=payload is not None, payload={"payment_status": payload or {}})

    def get_vehicle_details(vehicle_id: str) -> ToolResult:
        payload = repository.get_vehicle_details(vehicle_id=vehicle_id)
        return ToolResult(name="get_vehicle_details", success=payload is not None, payload={"vehicle_details": payload or {}})

    def get_ticket_history(user_id: str) -> ToolResult:
        payload = repository.get_ticket_history(user_id=user_id)
        return ToolResult(name="get_ticket_history", success=True, payload={"ticket_history": payload})

    def get_ticket_details(ticket_id: str) -> ToolResult:
        payload = repository.get_ticket_details(ticket_id=ticket_id)
        return ToolResult(name="get_ticket_details", success=payload is not None, payload={"ticket_details": payload or {}})

    def get_ticket_comments(ticket_id: str) -> ToolResult:
        payload = repository.get_ticket_comments(ticket_id=ticket_id)
        return ToolResult(name="get_ticket_comments", success=True, payload={"ticket_comments": payload})

    def search_related_orders(user_id: str) -> ToolResult:
        payload = repository.search_related_orders(user_id=user_id)
        return ToolResult(name="search_related_orders", success=True, payload={"related_orders": payload})

    def get_user_enquiries(user_id: str, active_only: bool = True) -> ToolResult:
        payload = repository.get_user_enquiries(user_id=user_id, active_only=active_only)
        return ToolResult(name="get_user_enquiries", success=True, payload={"user_enquiries": payload})

    def get_ownership_record(order_id: str | None = None, user_id: str | None = None, vin: str | None = None) -> ToolResult:
        payload = repository.get_ownership_record(order_id=order_id, user_id=user_id, vin=vin)
        return ToolResult(name="get_ownership_record", success=payload is not None, payload={"ownership_record": payload or {}})

    def get_dealer_details(dealer_id: str | None = None, dealer_code: str | None = None) -> ToolResult:
        payload = repository.get_dealer_details(dealer_id=dealer_id, dealer_code=dealer_code)
        return ToolResult(name="get_dealer_details", success=payload is not None, payload={"dealer_details": payload or {}})

    def get_dealer_facility_details(facility_code: str | None = None, dealer_id: str | None = None) -> ToolResult:
        payload = repository.get_dealer_facility_details(facility_code=facility_code, dealer_id=dealer_id)
        return ToolResult(name="get_dealer_facility_details", success=True, payload={"dealer_facilities": payload})

    def get_test_ride_details(order_id: str | None = None, user_id: str | None = None, phone: str | None = None) -> ToolResult:
        payload = repository.get_test_ride_details(order_id=order_id, user_id=user_id, phone=phone)
        return ToolResult(name="get_test_ride_details", success=True, payload={"test_ride_details": payload})

    registry.register("get_user_profile", get_user_profile)
    registry.register("get_user_profile_by_mobile", get_user_profile_by_mobile)
    registry.register("get_booking_details", get_booking_details)
    registry.register("get_order_details", get_order_details)
    registry.register("get_payment_status", get_payment_status)
    registry.register("get_order_payment_status", get_order_payment_status)
    registry.register("get_vehicle_details", get_vehicle_details)
    registry.register("get_ticket_history", get_ticket_history)
    registry.register("get_ticket_details", get_ticket_details)
    registry.register("get_ticket_comments", get_ticket_comments)
    registry.register("search_related_orders", search_related_orders)
    registry.register("get_user_enquiries", get_user_enquiries)
    registry.register("get_ownership_record", get_ownership_record)
    registry.register("get_dealer_details", get_dealer_details)
    registry.register("get_dealer_facility_details", get_dealer_facility_details)
    registry.register("get_test_ride_details", get_test_ride_details)
    return registry
