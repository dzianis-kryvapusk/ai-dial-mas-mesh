from typing import Any

from task.tools.deployment.base_agent_tool import BaseAgentTool


class WebSearchAgentTool(BaseAgentTool):

    @property
    def deployment_name(self) -> str:
        return "web-search-agent"

    @property
    def name(self) -> str:
        return "web_search_agent"

    @property
    def description(self) -> str:
        return "Agent that can performs complex web search. Ask it to find some information, verify facts, or synthesize information from multiple sources"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The query or instruction to send to the WEB Search Agent."
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
