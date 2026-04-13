from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ConversationMessage(BaseModel):
    role: str
    content: str


class SupportTicketInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    ticket_id: str
    raw_user_message: str
    conversation_history: list[ConversationMessage] = Field(default_factory=list)
    user_id: str | None = None
    mobile: str | None = Field(default=None, validation_alias=AliasChoices("mobile", "moobile", "contact", "phone_number", "phone"))
    booking_id: str | None = None
    payment_id: str | None = None
    order_id: str | None = None
    order_number: str | None = Field(default=None, validation_alias=AliasChoices("order_number", "orderNumber"))
    vehicle_id: str | None = None
