from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any

from support_agent.schemas.tool import ToolResult


ToolFunction = Callable[..., ToolResult]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, name: str, tool_fn: ToolFunction) -> None:
        self._tools[name] = tool_fn

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def required_parameters(self, name: str) -> list[str]:
        signature = inspect.signature(self._tools[name])
        required: list[str] = []
        for parameter_name, parameter in signature.parameters.items():
            if parameter.default is inspect._empty:
                required.append(parameter_name)
        return required

    def run(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools[name]
        signature = inspect.signature(tool)
        accepted_arguments = {
            key: value
            for key, value in arguments.items()
            if key in signature.parameters
        }
        return tool(**accepted_arguments)
