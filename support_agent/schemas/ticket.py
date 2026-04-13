from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: str
    content: str


class SupportTicketInput(BaseModel):
    ticket_id: str
    raw_user_message: str
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    user_id: str | None = None
    booking_id: str | None = None
    payment_id: str | None = None
    order_id: str | None = None
    vehicle_id: str | None = None
