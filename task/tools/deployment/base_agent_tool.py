import json
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, Stage, Attachment
from pydantic import StrictStr

from task.tools.base_tool import BaseTool
from task.tools.models import ToolCallParams
from task.utils.stage import StageProcessor


class BaseAgentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        chunks = await self._call_model(tool_call_params)

        content = ''
        custom_content: CustomContent = CustomContent(attachments=[])
        stages_map: dict[int, Stage] = {}

        print("Reading chunks:")
        async for chunk in chunks:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    tool_call_params.stage.append_content(delta.content)
                    content += delta.content
                if cc := delta.custom_content:
                    if cc.attachments:
                        custom_content.attachments.extend(cc.attachments)

                    if cc.state:
                        custom_content.state = cc.state

                    cc_dict = cc.dict(exclude_none=True)
                    if stages := cc_dict.get("stages"):
                        for stg in stages:
                            idx = stg["index"]
                            if opened_stg := stages_map.get(idx):
                                if stg_name := stg.get("name"):
                                    opened_stg.append_name(stg_name)
                                elif stg_content := stg.get("content"):
                                    opened_stg.append_content(stg_content)
                                elif stg_attachments := stg.get("attachments"):
                                    for stg_attachment in stg_attachments:
                                        opened_stg.add_attachment(Attachment(**stg_attachment))
                                elif stg.get("status") and stg.get("status") == 'completed':
                                    StageProcessor.close_stage_safely(stages_map[idx])
                            else:
                                stages_map[idx] = StageProcessor.open_stage(tool_call_params.choice, stg.get("name"))
        print("=== Finish reading chunks ===")

        for stg in stages_map.values():
            StageProcessor.close_stage_safely(stg)

        for attachment in custom_content.attachments:
            tool_call_params.choice.add_attachment(
                Attachment(**attachment.dict(exclude_none=True))
            )

        return Message(
            role=Role.TOOL,
            content=StrictStr(content),
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )

    async def _call_model(self, tool_call_params: ToolCallParams):
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        if prompt := arguments.get("prompt"):
            tool_call_params.stage.append_name(f": {prompt}")
            # prompt will be in the last message
            del arguments["prompt"]

        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
            api_version='2025-01-01-preview'
        )
        return await client.chat.completions.create(
            messages=self._prepare_messages(tool_call_params),
            stream=True,
            deployment_name=self.deployment_name,
            extra_body={
                "custom_fields": {
                    "configuration": {**arguments}
                }
            },
            extra_headers={
                "x-conversation-id": tool_call_params.conversation_id,
            },
        )

    def _prepare_messages(self, tool_call_params: ToolCallParams) -> list[dict[str, Any]]:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        prompt = arguments["prompt"]
        propagate_history = bool(arguments.get("propagate_history", False))

        messages = []
        if propagate_history:
            for i, message in enumerate(tool_call_params.messages):
                if message.role == Role.ASSISTANT:
                    if message.custom_content and message.custom_content.state:
                        state = message.custom_content.state
                        if state.get(self.name):
                            # 1. add user request
                            messages.append(
                                # user message is always before assistant message
                                tool_call_params.messages[i - 1].dict(exclude_none=True)
                            )
                            # 2. restore appropriate format of assistant message
                            copied_msg = deepcopy(message)
                            copied_msg.custom_content.state = state.get(self.name)
                            messages.append(copied_msg.dict(exclude_none=True))

        custom_content = tool_call_params.messages[-1].custom_content
        messages.append(
            {
                "role": "user",
                "content": prompt,
                "custom_content": custom_content.dict(exclude_none=True) if custom_content else None,
            }
        )

        return messages