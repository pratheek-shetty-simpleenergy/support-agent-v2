from __future__ import annotations

from pydantic import BaseModel, Field


class TableBinding(BaseModel):
    schema_name: str = "public"
    table: str
    description: str
    columns: list[str] = Field(default_factory=list)


class DatabaseBinding(BaseModel):
    database_name: str
    description: str
    tables: dict[str, TableBinding]


def default_business_db_catalog() -> dict[str, DatabaseBinding]:
    return {
        "dms_stage": DatabaseBinding(
            database_name="dms-stage",
            description="Dealer and dealership details.",
            tables={
                "dealers": TableBinding(
                    table="dealer",
                    description="Dealer master records and dealership details.",
                    columns=["id", "name", "email", "phone_number", "authorised_dealer", "serial_number", "created_at", "updated_at", "primary_contact_name", "primary_contact_number", "primary_contact_email", "description", "status", "role_id", "role_name"],
                ),
                "dealer_facilities": TableBinding(
                    table="dealer_facility_map",
                    description="Dealer facilities such as showroom, store, and service center details.",
                    columns=["id", "dealer_id", "type", "location", "address", "city", "state", "pincode", "latitude", "longitude", "status", "store_timings", "facility_code"],
                ),
                "dealer_members": TableBinding(
                    table="dealer_member",
                    description="Dealer staff and facility-level team members.",
                    columns=["id", "role_id", "role_name", "phone_number", "name", "email", "status", "created_at", "updated_at", "description", "facility_id"],
                ),
            },
        ),
        "ownership_stage": DatabaseBinding(
            database_name="ownership-stage",
            description="Ownership details including VIN, order mapping, user mapping, and registration details.",
            tables={
                "ownerships": TableBinding(
                    table='"Ownership"',
                    description="Ownership facts linking VIN, order, user, and registration details.",
                    columns=["id", '"userId"', "vin", '"startDate"', '"endDate"', "metadata", '"createdAt"', '"updatedAt"', '"orderId"', '"registrationNumber"'],
                ),
                "ownership_transfers": TableBinding(
                    table='"OwnershipTransfer"',
                    description="Ownership transfer requests and approval status.",
                    columns=["id", "vin", '"oldUserId"', '"newUserId"', '"transferStatus"', '"requestedBy"', '"adminId"', "metadata", '"createdAt"', '"updatedAt"'],
                ),
                "documents": TableBinding(
                    table='"Documents"',
                    description="Ownership-related documents for VIN, order, or user.",
                    columns=["id", "type", "name", "thumbnail", "metadata", '"createdAt"', '"updatedAt"', '"userId"', '"orderId"', "vin", '"viewType"', '"s3Key"'],
                ),
            },
        ),
        "orders_stage": DatabaseBinding(
            database_name="orders-stage",
            description="Order details, booking status, payment transactions, and refund history.",
            tables={
                "orders": TableBinding(
                    table='"Order"',
                    description="Order state and order-level transaction details.",
                    columns=["id", '"orderNumber"', '"variantId"', '"userId"', "pincode", "city", "amount", "status", '"statusMessage"', '"paymentMetadata"', '"orderMetadata"', '"createdAt"', '"updatedAt"', '"enquirySfId"', '"orderSource"', "rsa", '"facilityId"', '"tradeIn"', "financing", '"attendedBy"'],
                ),
                "enquiries": TableBinding(
                    table='"Enquiry"',
                    description="Order enquiry lifecycle and follow-up state before order conversion.",
                    columns=["id", '"orderNumber"', '"variantId"', '"userId"', "pincode", "city", "amount", '"enquiryStatus"', '"statusMessage"', '"facilityId"', '"orderMetadata"', '"createdAt"', '"updatedAt"', '"enquirySfId"', '"orderSource"', '"paymentSessionId"', "rsa", '"tradeIn"', "financing", '"followUpStage"', '"lastFollowUpAt"', '"nextFollowUpAt"', '"qualificationStatus"', '"attendedBy"'],
                ),
                "transactions": TableBinding(
                    table='"Transaction"',
                    description="Payment transaction records linked to orders.",
                    columns=["id", '"orderId"', '"transactionId"', "amount", '"paymentStatus"', '"transactionMetadata"', '"createdAt"', '"updatedAt"'],
                ),
                "refunds": TableBinding(
                    table='"Refund"',
                    description="Refund history and refund lifecycle details.",
                    columns=["id", '"orderId"', "reason", '"initiatedBy"', '"userId"', '"initiatedOn"', "status", '"refundMetadata"', '"expectedAt"', '"createdAt"', '"updatedAt"', '"additionalComment"'],
                ),
                "payment_links": TableBinding(
                    table='"PaymentLink"',
                    description="Payment link lifecycle for enquiries.",
                    columns=["id", '"enquiryId"', '"linkId"', '"cfLinkId"', '"linkUrl"', '"linkStatus"', '"linkAmount"', '"linkAmountPaid"', '"linkCurrency"', '"linkExpiryTime"', '"customerDetails"', '"linkNotes"', '"linkNotify"', '"payloadSnapshot"', '"createdAt"', '"updatedAt"'],
                ),
            },
        ),
        "testride_stage": DatabaseBinding(
            database_name="testride-stage",
            description="Test ride bookings, media, and feedback details.",
            tables={
                "test_rides": TableBinding(
                    table='"TestRideData"',
                    description="Test ride booking data keyed by order, user, phone, and facility.",
                    columns=["id", '"orderId"', "source", "date", "time", "metadata", '"createdAt"', '"updatedAt"', "phone", "pincode", '"facilityId"', '"drivingLicenseNumber"', '"userId"', "status", '"testRideType"', '"homeTestRideAddressData"'],
                ),
                "feedback": TableBinding(
                    table='"FeedbackData"',
                    description="Feedback records tied to test rides.",
                    columns=["id", '"orderId"', '"testRideId"', '"createdAt"', '"updatedAt"'],
                ),
                "responses": TableBinding(
                    table='"Response"',
                    description="Question and response entries for test ride feedback.",
                    columns=["id", "type", "header", "text", "response", '"feedbackId"', '"createdAt"', '"updatedAt"'],
                ),
            },
        ),
        "users_stage": DatabaseBinding(
            database_name="users-stage",
            description="User profile data such as mobile number, email, and customer identity details.",
            tables={
                "users": TableBinding(
                    table='"User"',
                    description="User master profile records.",
                    columns=["id", "name", "mobile", "email", '"primaryVin"', '"profilePicture"', '"signupSource"', '"emailVerified"', '"whatsappConsent"', '"userMetadata"', '"createdAt"', '"updatedAt"'],
                ),
                "addresses": TableBinding(
                    table='"Address"',
                    description="User addresses and saved address metadata.",
                    columns=["id", '"addressHeader"', '"addressDetails"', "type", "eloc", '"addressMetadata"', '"createdAt"', '"updatedAt"', "pincode", '"userID"', '"isFavourite"'],
                ),
                "documents": TableBinding(
                    table='"Document"',
                    description="User documents such as ID and vehicle-related uploads.",
                    columns=["id", "vin", "type", "name", "link", "thumbnail", '"documentMetadata"', '"createdAt"', '"updatedAt"', '"userID"'],
                ),
                "emergency_contacts": TableBinding(
                    table='"EmergencyContact"',
                    description="Emergency contacts linked to a user.",
                    columns=["id", '"userID"', "name", "phone", '"isPrimary"', '"createdAt"', '"updatedAt"'],
                ),
            },
        ),
        "unified_ticketing_stage": DatabaseBinding(
            database_name="unified-ticketing-stage",
            description="Support tickets and support investigation history.",
            tables={
                "tickets": TableBinding(
                    table='"Ticket"',
                    description="Support ticket records and historical ticket state.",
                    columns=["id", '"ticketCode"', '"userId"', "vin", '"phoneNumber"', '"assignedAgentId"', "category", "subcategory", "reason", "status", "priority", "source", '"resolutionSummary"', '"notificationSent"', '"createdAt"', '"updatedAt"', '"closedAt"', "name", "email", '"assignedEngineeringTeamId"', '"assignedEngineeringMemberId"', "disposition", '"subDisposition"', '"resolvedAt"', '"resolvedByAgentId"', '"resolvedByEngineeringMemberId"', '"resolvedByEngineeringTeamId"'],
                ),
                "ticket_comments": TableBinding(
                    table='"TicketComment"',
                    description="Ticket conversation comments by agent, customer, or system.",
                    columns=["id", '"ticketId"', '"authorId"', '"authorRole"', '"commentText"', '"commentType"', '"createdAt"'],
                ),
                "ticket_actions": TableBinding(
                    table='"TicketAction"',
                    description="Recorded actions taken during ticket handling.",
                    columns=["id", '"ticketId"', '"actionTakenBy"', '"actionDescription"', '"actionStatus"', '"executedAt"'],
                ),
                "ticket_status_history": TableBinding(
                    table='"TicketStatusHistory"',
                    description="Ticket status changes over time.",
                    columns=["id", '"ticketId"', '"fromStatus"', '"toStatus"', '"changedBy"', "notes", '"createdAt"'],
                ),
                "ticket_assignment_history": TableBinding(
                    table='"TicketAssignmentHistory"',
                    description="Ticket assignment changes across agents and engineering teams.",
                    columns=["id", '"ticketId"', '"fromAgentId"', '"toAgentId"', '"fromEngineeringTeamId"', '"toEngineeringTeamId"', '"fromEngineeringMemberId"', '"toEngineeringMemberId"', '"assignedById"', '"assignedByRole"', "reason", '"createdAt"'],
                ),
            },
        ),
    }
