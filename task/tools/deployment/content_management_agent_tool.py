from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class ContentManagementAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        return "content-management-agent"

    @property
    def name(self) -> str:
        return "content_management_agent"

    @property
    def description(self) -> str:
        return (
            "Agent that can work with files. Ask it when you need to:\n"
            " - extract and analyze files content.\n"
            " - performs RAG Search through files content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The query or instruction to send to the Content Management Agent."
                },
                "propagate_history": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Flag to enable including the previous conversation history or not.\n"
                        "If true - all the previous messages will be sent for context continuity.\n"
                        "If false - send an individual message without historical context.\n"
                        "Notes:\n"
                        " - Only the conversation history between these two agents is shared; interactions with other agents are never included.\n"
                        " - Should be set to `true` only when the `prompt` lacks sufficient context and the required context exists in the conversation history.")
                },
            },
            "required": [
                "prompt"
            ]
        }
