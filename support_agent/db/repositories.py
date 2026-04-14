from __future__ import annotations

import re

from support_agent.config import Settings
from support_agent.db.catalog import TableBinding
from support_agent.db.client import BusinessDbManager


class BusinessDbRepository:
    def __init__(self, db: BusinessDbManager, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.catalog = settings.business_db_catalog

    def get_user_profile(self, user_id: str) -> dict | None:
        binding = self._table("users_stage", "users")
        sql = f"""
        SELECT
          "id" AS id,
          "name" AS name,
          "mobile" AS mobile,
          "email" AS email,
          "primaryVin" AS primary_vin,
          "profilePicture" AS profile_picture,
          "signupSource" AS signup_source,
          "emailVerified" AS email_verified,
          "whatsappConsent" AS whatsapp_consent,
          "userMetadata" AS user_metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "id" = :user_id
        LIMIT 1
        """
        return self.db.fetch_one(sql, {"user_id": user_id}, database_key="users_stage")

    def get_user_profile_by_mobile(self, mobile: str) -> dict | None:
        binding = self._table("users_stage", "users")
        normalized_mobile = _normalize_mobile_number(mobile)
        sql = f"""
        SELECT
          "id" AS id,
          "name" AS name,
          "mobile" AS mobile,
          "email" AS email,
          "primaryVin" AS primary_vin,
          "profilePicture" AS profile_picture,
          "signupSource" AS signup_source,
          "emailVerified" AS email_verified,
          "whatsappConsent" AS whatsapp_consent,
          "userMetadata" AS user_metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "mobile" = :mobile
        LIMIT 1
        """
        return self.db.fetch_one(sql, {"mobile": normalized_mobile}, database_key="users_stage")

    def get_booking_details(self, booking_id: str) -> dict | None:
        return self.get_order_details(
            order_id=booking_id if _is_uuid_like(booking_id) else None,
            order_number=None if _is_uuid_like(booking_id) else booking_id,
        )

    def get_order_details(self, order_id: str | None = None, order_number: str | None = None) -> dict | None:
        binding = self._table("orders_stage", "orders")
        select_sql = f"""
        SELECT
          "id" AS id,
          "orderNumber" AS order_number,
          "userId" AS user_id,
          "variantId" AS variant_id,
          "status" AS status,
          "statusMessage" AS status_message,
          "amount" AS amount,
          "city" AS city,
          "pincode" AS pincode,
          "paymentMetadata" AS payment_metadata,
          "orderMetadata" AS order_metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        """
        if order_id:
            sql = f'{select_sql} WHERE "id" = :order_id LIMIT 1'
            return self.db.fetch_one(sql, {"order_id": order_id}, database_key="orders_stage")
        if order_number:
            sql = f'{select_sql} WHERE "orderNumber" = :order_number LIMIT 1'
            return self.db.fetch_one(sql, {"order_number": order_number}, database_key="orders_stage")
        raise ValueError("order_id or order_number is required for order lookup.")

    def get_payment_status(self, payment_id: str) -> dict | None:
        binding = self._table("orders_stage", "transactions")
        select_sql = f"""
        SELECT
          "id" AS id,
          "orderId" AS order_id,
          "transactionId" AS transaction_id,
          "amount" AS amount,
          "paymentStatus" AS payment_status,
          "transactionMetadata" AS transaction_metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        """
        if _is_uuid_like(payment_id):
            sql = f'{select_sql} WHERE "id" = :payment_id LIMIT 1'
        else:
            sql = f'{select_sql} WHERE "transactionId" = :payment_id LIMIT 1'
        return self.db.fetch_one(sql, {"payment_id": payment_id}, database_key="orders_stage")

    def get_order_payment_status(self, order_id: str) -> dict | None:
        binding = self._table("orders_stage", "transactions")
        sql = f"""
        SELECT
          "id" AS id,
          "orderId" AS order_id,
          "transactionId" AS transaction_id,
          "amount" AS amount,
          "paymentStatus" AS payment_status,
          "transactionMetadata" AS transaction_metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "orderId" = :order_id
        ORDER BY "createdAt" DESC
        LIMIT 1
        """
        return self.db.fetch_one(sql, {"order_id": order_id}, database_key="orders_stage")

    def get_vehicle_details(self, vehicle_id: str) -> dict | None:
        binding = self._table("ownership_stage", "ownerships")
        select_sql = f"""
        SELECT
          "id" AS id,
          "userId" AS user_id,
          "vin" AS vin,
          "orderId" AS order_id,
          "registrationNumber" AS registration_number,
          "startDate" AS start_date,
          "endDate" AS end_date,
          "metadata" AS metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        """
        if _is_uuid_like(vehicle_id):
            sql = f'{select_sql} WHERE "id" = :vehicle_id LIMIT 1'
        else:
            sql = f'{select_sql} WHERE "vin" = :vehicle_id OR "registrationNumber" = :vehicle_id LIMIT 1'
        return self.db.fetch_one(sql, {"vehicle_id": vehicle_id}, database_key="ownership_stage")

    def get_ticket_history(self, user_id: str) -> list[dict]:
        binding = self._table("unified_ticketing_stage", "tickets")
        sql = f"""
        SELECT
          "id" AS id,
          "ticketCode" AS ticket_code,
          "userId" AS user_id,
          "vin" AS vin,
          "phoneNumber" AS phone_number,
          "category" AS category,
          "subcategory" AS subcategory,
          "reason" AS reason,
          "status" AS status,
          "priority" AS priority,
          "source" AS source,
          "resolutionSummary" AS resolution_summary,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at,
          "closedAt" AS closed_at,
          "disposition" AS disposition,
          "subDisposition" AS sub_disposition,
          "resolvedAt" AS resolved_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "userId" = :user_id
        ORDER BY "createdAt" DESC
        LIMIT 10
        """
        return self.db.fetch_all(sql, {"user_id": user_id}, database_key="unified_ticketing_stage")

    def get_ticket_details(self, ticket_id: str) -> dict | None:
        binding = self._table("unified_ticketing_stage", "tickets")
        sql = f"""
        SELECT
          "id" AS id,
          "ticketCode" AS ticket_code,
          "userId" AS user_id,
          "vin" AS vin,
          "phoneNumber" AS phone_number,
          "assignedAgentId" AS assigned_agent_id,
          "category" AS category,
          "subcategory" AS subcategory,
          "reason" AS reason,
          "status" AS status,
          "priority" AS priority,
          "source" AS source,
          "resolutionSummary" AS resolution_summary,
          "notificationSent" AS notification_sent,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at,
          "closedAt" AS closed_at,
          "name" AS name,
          "email" AS email,
          "assignedEngineeringTeamId" AS assigned_engineering_team_id,
          "assignedEngineeringMemberId" AS assigned_engineering_member_id,
          "disposition" AS disposition,
          "subDisposition" AS sub_disposition,
          "resolvedAt" AS resolved_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "id" = :ticket_id
           OR "ticketCode" = :ticket_id
        LIMIT 1
        """
        return self.db.fetch_one(sql, {"ticket_id": ticket_id}, database_key="unified_ticketing_stage")

    def get_ticket_comments(self, ticket_id: str) -> list[dict]:
        binding = self._table("unified_ticketing_stage", "ticket_comments")
        sql = f"""
        SELECT
          "id" AS id,
          "ticketId" AS ticket_id,
          "authorId" AS author_id,
          "authorRole" AS author_role,
          "commentText" AS comment_text,
          "commentType" AS comment_type,
          "createdAt" AS created_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "ticketId" = :ticket_id
        ORDER BY "createdAt" DESC
        LIMIT 20
        """
        return self.db.fetch_all(sql, {"ticket_id": ticket_id}, database_key="unified_ticketing_stage")

    def search_related_orders(self, user_id: str) -> list[dict]:
        binding = self._table("orders_stage", "orders")
        sql = f"""
        SELECT
          "id" AS id,
          "orderNumber" AS order_number,
          "userId" AS user_id,
          "status" AS status,
          "amount" AS amount,
          "city" AS city,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "userId" = :user_id
        ORDER BY "createdAt" DESC
        LIMIT 10
        """
        return self.db.fetch_all(sql, {"user_id": user_id}, database_key="orders_stage")

    def get_user_enquiries(self, user_id: str, active_only: bool = True) -> list[dict]:
        binding = self._table("orders_stage", "enquiries")
        status_filter = 'AND "enquiryStatus" = \'ACTIVE\'' if active_only else ""
        sql = f"""
        SELECT
          "id" AS id,
          "orderNumber" AS order_number,
          "userId" AS user_id,
          "enquiryStatus" AS enquiry_status,
          "statusMessage" AS status_message,
          "amount" AS amount,
          "city" AS city,
          "paymentSessionId" AS payment_session_id,
          "followUpStage" AS follow_up_stage,
          "lastFollowUpAt" AS last_follow_up_at,
          "nextFollowUpAt" AS next_follow_up_at,
          "qualificationStatus" AS qualification_status,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        WHERE "userId" = :user_id
        {status_filter}
        ORDER BY "createdAt" DESC
        LIMIT 10
        """
        return self.db.fetch_all(sql, {"user_id": user_id}, database_key="orders_stage")

    def get_ownership_record(self, order_id: str | None = None, user_id: str | None = None, vin: str | None = None) -> dict | None:
        binding = self._table("ownership_stage", "ownerships")
        params: dict[str, object] = {}
        filters: list[str] = []

        if order_id:
            filters.append('"orderId" = :order_id')
            params["order_id"] = order_id
        if user_id:
            filters.append('"userId" = :user_id')
            params["user_id"] = user_id
        if vin:
            filters.append('"vin" = :vin')
            params["vin"] = vin

        if not filters:
            raise ValueError("At least one of order_id, user_id, or vin is required for ownership lookup.")

        sql = f"""
        SELECT
          "id" AS id,
          "userId" AS user_id,
          "vin" AS vin,
          "orderId" AS order_id,
          "registrationNumber" AS registration_number,
          "startDate" AS start_date,
          "endDate" AS end_date,
          "metadata" AS metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at
        FROM {binding.schema_name}.{binding.table}
        WHERE {" OR ".join(filters)}
        LIMIT 1
        """
        return self.db.fetch_one(sql, params, database_key="ownership_stage")

    def get_dealer_details(self, dealer_id: str | None = None, dealer_code: str | None = None) -> dict | None:
        binding = self._table("dms_stage", "dealers")
        if dealer_id:
            sql = f"""
            SELECT
              id,
              name,
              email,
              phone_number,
              authorised_dealer,
              serial_number,
              created_at,
              updated_at,
              primary_contact_name,
              primary_contact_number,
              primary_contact_email,
              description,
              status,
              role_id,
              role_name
            FROM {binding.schema_name}.{binding.table}
            WHERE id = :dealer_id
            LIMIT 1
            """
            return self.db.fetch_one(sql, {"dealer_id": dealer_id}, database_key="dms_stage")
        if dealer_code:
            sql = f"""
            SELECT
              d.id,
              d.name,
              d.email,
              d.phone_number,
              d.authorised_dealer,
              d.serial_number,
              d.created_at,
              d.updated_at,
              d.primary_contact_name,
              d.primary_contact_number,
              d.primary_contact_email,
              d.description,
              d.status,
              d.role_id,
              d.role_name
            FROM {binding.schema_name}.{binding.table} d
            JOIN {binding.schema_name}.dealer_facility_map dfm
              ON dfm.dealer_id = d.id
            WHERE dfm.facility_code = :dealer_code
            LIMIT 1
            """
            return self.db.fetch_one(sql, {"dealer_code": dealer_code}, database_key="dms_stage")
        raise ValueError("dealer_id or dealer_code is required for dealer lookup.")

    def get_dealer_facility_details(self, facility_code: str | None = None, dealer_id: str | None = None) -> list[dict]:
        binding = self._table("dms_stage", "dealer_facilities")
        if not facility_code and not dealer_id:
            raise ValueError("facility_code or dealer_id is required for dealer facility lookup.")

        params: dict[str, object] = {}
        filters: list[str] = []
        if facility_code:
            filters.append("facility_code = :facility_code")
            params["facility_code"] = facility_code
        if dealer_id:
            filters.append("dealer_id = :dealer_id")
            params["dealer_id"] = dealer_id

        sql = f"""
        SELECT
          id,
          dealer_id,
          type,
          location,
          address,
          city,
          state,
          pincode,
          latitude,
          longitude,
          status,
          store_timings,
          facility_code
        FROM {binding.schema_name}.{binding.table}
        WHERE {" OR ".join(filters)}
        ORDER BY facility_code ASC
        LIMIT 10
        """
        return self.db.fetch_all(sql, params, database_key="dms_stage")

    def get_test_ride_details(
        self,
        order_id: str | None = None,
        user_id: str | None = None,
        phone: str | None = None,
    ) -> list[dict]:
        binding = self._table("testride_stage", "test_rides")
        params: dict[str, object] = {}
        filters: list[str] = []
        if order_id:
            filters.append('"orderId" = :order_id')
            params["order_id"] = order_id
        if user_id:
            filters.append('"userId" = :user_id')
            params["user_id"] = user_id
        if phone:
            filters.append('phone = :phone')
            params["phone"] = phone
        if not filters:
            raise ValueError("order_id, user_id, or phone is required for test ride lookup.")

        sql = f"""
        SELECT
          "id" AS id,
          "orderId" AS order_id,
          "source" AS source,
          "date" AS date,
          "time" AS time,
          "metadata" AS metadata,
          "createdAt" AS created_at,
          "updatedAt" AS updated_at,
          "phone" AS phone,
          "pincode" AS pincode,
          "facilityId" AS facility_id,
          "drivingLicenseNumber" AS driving_license_number,
          "userId" AS user_id,
          "status" AS status,
          "testRideType" AS test_ride_type,
          "homeTestRideAddressData" AS home_test_ride_address_data
        FROM {binding.schema_name}.{binding.table}
        WHERE {" OR ".join(filters)}
        ORDER BY "createdAt" DESC
        LIMIT 10
        """
        return self.db.fetch_all(sql, params, database_key="testride_stage")

    def _table(self, database_key: str, table_key: str) -> TableBinding:
        return self.catalog[database_key].tables[table_key]


def _normalize_mobile_number(mobile: str) -> str:
    digits = "".join(character for character in mobile if character.isdigit())
    if digits.startswith("91") and len(digits) == 12:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+91{digits}"
    if mobile.startswith("+"):
        return mobile
    return mobile


def _is_uuid_like(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", value))
