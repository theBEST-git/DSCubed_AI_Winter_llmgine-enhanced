import uuid
import json
import asyncio
from typing import Any

from llmgine.bus.bus import MessageBus
from llmgine.llm.context.memory import SimpleChatHistory
from llmgine.llm.models.openai_models import Gpt41Mini
from llmgine.llm.providers.providers import Providers
from llmgine.llm.tools.tool_manager import ToolManager
from llmgine.llm.tools import ToolCall
from llmgine.llm.models.openai_models import OpenAIResponse
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from llmgine.messages.commands import Command, CommandResult
from llmgine.messages.events import Event
from llmgine.ui.cli.cli import EngineCLI
from llmgine.ui.cli.components import EngineResultComponent
from dataclasses import dataclass
from llmgine.llm import SessionID, AsyncOrSyncToolFunction

from llmgine.settings import MAX_CONTEXT_TOKENS, PRUNE_TARGET_TOKENS
from llmgine.utils.token_count import count_tokens


@dataclass
class ToolChatEngineCommand(Command):
    """Command for the Tool Chat Engine."""

    prompt: str = ""


@dataclass
class ToolChatEngineStatusEvent(Event):
    """Event emitted when the status of the engine changes."""

    status: str = ""


@dataclass
class ToolChatEngineToolResultEVent(Event):
    """Event emitted when a tool is executed."""

    tool_name: str = ""
    result: Any = None

@dataclass
class MemorySummarisedEvent(Event):
    """
    Fired when older messages are condensed into a summary.
    """
    removed_tokens: int = 0
    summary_tokens: int = 0

class ToolChatEngine:
    def __init__(
        self,
        session_id: SessionID,
    ):
        """Initialize the LLM engine.

        Args:
            session_id: The session identifier
        """
        # Use the provided message bus or create a new one
        self.message_bus: MessageBus = MessageBus()
        self.engine_id: str = str(uuid.uuid4())
        self.session_id: SessionID = SessionID(session_id)

        # Create tightly coupled components - pass the simple engine
        self.context_manager = SimpleChatHistory(
            engine_id=self.engine_id, session_id=self.session_id
        )
        self.llm_manager = Gpt41Mini(Providers.OPENAI)
        self.tool_manager = ToolManager(
            engine_id=self.engine_id, session_id=self.session_id, llm_model_name="openai"
        )

    # ---------------------------------------------------------------------
    # Internals for summarisation / pruning
    # ---------------------------------------------------------------------

    def _get_chat_history(self) -> list[dict]:
        """Direct access to SimpleChatHistory.chat_history (excluding system prompt)."""
        return self.context_manager.chat_history

    def _replace_slice_with_summary(self, start: int, end: int, summary: str) -> None:
        """
        Replace messages[start:end] with a single assistant summary message.
        """
        self._get_chat_history()[start:end] = [
            {"role": "assistant", "name": "summary", "content": summary}
        ]

    # -----------------new functions added ↑↑↑-----------------------------

    async def handle_command(self, command: ToolChatEngineCommand) -> CommandResult:
        """Handle a prompt command following OpenAI tool usage pattern.

        Args:
            command: The prompt command to handle

        Returns:
            CommandResult: The result of the command execution
        """
        try:
            # 1. Add user message to history
            self.context_manager.store_string(command.prompt, "user")

            # Loop for potential tool execution cycles
            while True:
                # NEW: prune/summarise if needed
                await self._maybe_summarise_history()

                # 2. Get current context (including latest user message or tool results)
                current_context = await self.context_manager.retrieve()

                # 3. Get available tools
                tools = await self.tool_manager.get_tools()

                # 4. Call LLM
                await self.message_bus.publish(
                    ToolChatEngineStatusEvent(
                        status="calling LLM", session_id=self.session_id
                    )
                )
                response: OpenAIResponse = await self.llm_manager.generate(
                    messages=current_context, tools=tools
                )
                assert isinstance(response, OpenAIResponse), (
                    "response is not an OpenAIResponse"
                )

                # 5. Extract the first choice's message object
                response_message: ChatCompletionMessage = response.raw.choices[0].message
                assert isinstance(response_message, ChatCompletionMessage), (
                    "response_message is not a ChatCompletionMessage"
                )

                # 6. Add the entire assistant message object to history
                await self.context_manager.store_assistant_message(response_message)

                # 7. Check for tool calls
                if not response_message.tool_calls:
                    # No tool calls, break the loop and return the content
                    final_content = response_message.content or ""

                    # Notify status complete
                    await self.message_bus.publish(
                        ToolChatEngineStatusEvent(
                            status="finished", session_id=self.session_id
                        )
                    )
                    return CommandResult(
                        success=True, result=final_content, session_id=self.session_id
                    )

                # 8. Process tool calls
                for tool_call in response_message.tool_calls:
                    tool_call_obj = ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    try:
                        # Execute the tool
                        await self.message_bus.publish(
                            ToolChatEngineStatusEvent(
                                status="executing tool", session_id=self.session_id
                            )
                        )

                        result = await self.tool_manager.execute_tool_call(tool_call_obj)

                        # Convert result to string if needed for history
                        if isinstance(result, dict):
                            result_str = json.dumps(result)
                        else:
                            result_str = str(result)

                        # Store tool execution result in history
                        self.context_manager.store_tool_call_result(
                            tool_call_id=tool_call_obj.id,
                            name=tool_call_obj.name,
                            content=result_str,
                        )

                        # Publish tool execution event
                        await self.message_bus.publish(
                            ToolChatEngineToolResultEVent(
                                tool_name=tool_call_obj.name,
                                result=result_str,
                                session_id=self.session_id,
                            )
                        )

                    except Exception as e:
                        error_msg = f"Error executing tool {tool_call_obj.name}: {str(e)}"
                        print(error_msg)
                        # Store error result in history
                        self.context_manager.store_tool_call_result(
                            tool_call_id=tool_call_obj.id,
                            name=tool_call_obj.name,
                            content=error_msg,
                        )
                # After processing all tool calls, loop back to call the LLM again with the updated context.

        except Exception as e:
            print(f"ERROR in handle_prompt_command: {e}")
            import traceback

            traceback.print_exc()

            return CommandResult(success=False, error=str(e), session_id=self.session_id)

    async def register_tool(self, function: AsyncOrSyncToolFunction):
        """Register a function as a tool."""
        await self.tool_manager.register_tool(function)
        print(f"Tool registered: {function.__name__}")

    async def clear_context(self):
        """Clear the conversation context."""
        self.context_manager.clear()

    def set_system_prompt(self, prompt: str):
        """Set the system prompt."""
        self.context_manager.set_system_prompt(prompt)

    async def _maybe_summarise_history(self) -> None:
        """
        If the total context exceeds MAX_CONTEXT_TOKENS, condense the oldest
        PRUNE_TARGET_TOKENS into a short summary to keep the chat lean.
        """
        # Build the outgoing prompt (system + history) to measure size
        messages = await self.context_manager.retrieve()
        total_tokens = count_tokens(messages)

        if total_tokens <= MAX_CONTEXT_TOKENS:
            return  # still small, nothing to do

        # -----------------------------------------------------------------
        # 1. Determine slice of chat_history to summarise
        #    chat_history excludes system prompt, so indices map directly.
        # -----------------------------------------------------------------
        slice_tokens = 0
        slice_end = 0
        history = self._get_chat_history()
        for i, msg in enumerate(history):
            slice_tokens += len(msg.get("content") or "")
            if slice_tokens >= PRUNE_TARGET_TOKENS:
                slice_end = i + 1
                break

        slice_to_summarise = history[:slice_end]

        # -----------------------------------------------------------------
        # 2. Ask the LLM to condense the slice
        # -----------------------------------------------------------------
        summarise_prompt = [
            {"role": "system",
            "content": "Summarise the following conversation in one concise paragraph, \
                        preserving key facts, decisions, and named entities."},
            *slice_to_summarise,
        ]
        summary_resp = await self.llm_manager.generate(messages=summarise_prompt)
        summary_text = summary_resp.raw.choices[0].message.content.strip()

        # -----------------------------------------------------------------
        # 3. Replace old slice with summary & publish event
        # -----------------------------------------------------------------
        removed_tokens = count_tokens(slice_to_summarise)
        self._replace_slice_with_summary(0, slice_end, summary_text)

        await self.message_bus.publish(
            MemorySummarisedEvent(
                removed_tokens=removed_tokens,
                summary_tokens=count_tokens([{"content": summary_text}]),
                session_id=self.session_id,
            )
        )

        print("Messages Condensation Finished 🛠")



async def main():
    import os

    print(f"Current working directory: {os.getcwd()}")

    from tools.test_tools import get_weather
    from llmgine.ui.cli.components import ToolComponent
    from llmgine.bootstrap import ApplicationBootstrap, ApplicationConfig

    config = ApplicationConfig(enable_console_handler=False)
    bootstrap = ApplicationBootstrap(config)
    await bootstrap.bootstrap()

    cli = EngineCLI(SessionID("test"))
    engine = ToolChatEngine(session_id=SessionID("test"))
    await engine.register_tool(get_weather)
    cli.register_engine(engine)
    cli.register_engine_command(ToolChatEngineCommand, engine.handle_command)
    cli.register_engine_result_component(EngineResultComponent)
    cli.register_loading_event(ToolChatEngineStatusEvent)
    cli.register_component_event(ToolChatEngineToolResultEVent, ToolComponent)
    await cli.main()


if __name__ == "__main__":
    asyncio.run(main())
