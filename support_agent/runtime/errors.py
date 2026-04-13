from __future__ import annotations


class SupportAgentError(RuntimeError):
    error_type = "support_agent_error"
    retryable = False


class TransientDependencyError(SupportAgentError):
    error_type = "transient_dependency_error"
    retryable = True


class PermanentDependencyError(SupportAgentError):
    error_type = "permanent_dependency_error"


class InvalidModelOutputError(SupportAgentError):
    error_type = "invalid_model_output"


class ToolExecutionError(SupportAgentError):
    error_type = "tool_execution_error"


class ValidationRuntimeError(SupportAgentError):
    error_type = "validation_error"


class UserClarificationRequired(SupportAgentError):
    error_type = "user_clarification_required"


class EscalationRequired(SupportAgentError):
    error_type = "escalation_required"

