#!/usr/bin/env python
import asyncio
import logging
import os
from typing import Callable, Dict, List, Tuple

import chainlit as cl
from dotenv import load_dotenv
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.agent import AgentRunner
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.ollama import OllamaEmbedding
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
from rich.logging import RichHandler
from rich.traceback import install

# https://rich.readthedocs.io/en/latest/logging.html#handle-exceptions
logging.basicConfig(
    # level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger()

# https://rich.readthedocs.io/en/stable/traceback.html#traceback-handler
install(show_locals=True)


tracer_provider = register(
    project_name="besties",
    endpoint="http://localhost:6006/v1/traces",
)


LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

user_name = os.getlogin()

# https://rich.readthedocs.io/en/latest/logging.html#handle-exceptions
logging.basicConfig(
    # level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger()

# https://rich.readthedocs.io/en/stable/traceback.html#traceback-handler
install(show_locals=True)


# ruff: noqa: E402
# Keep this here to ensure imports have environment available.
env_found = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

shared_states = {"is_conversation_going": False}


def create_callback_manager(should_use_chainlit: bool = False) -> CallbackManager:
    debug_logger = logging.getLogger("debug")
    debug_logger.setLevel(logging.DEBUG)
    callback_handlers = [
        LlamaDebugHandler(logger=debug_logger),
    ]
    if should_use_chainlit:
        callback_handlers.append(cl.LlamaIndexCallbackHandler())
    return CallbackManager(callback_handlers)


def set_up_llama_index(
    should_use_chainlit: bool = False,
):
    """
    One-time setup code for shared objects across all AgentRunners.
    """
    # Needed for "Retrieved the following sources" to show up on Chainlit.
    Settings.callback_manager = create_callback_manager(should_use_chainlit)
    # ============= Beginning of the code block for wiring on to models. =============
    # At least when Chainlit is involved, LLM initializations must happen upon the `@cl.on_chat_start` event,
    # not in the global scope.
    # Otherwise, it messes up with Arize Phoenix: LLM calls won't be captured as parts of an Agent Step.
    if api_key := os.environ.get("OPENAI_API_KEY", None):
        logger.info("Using OpenAI API.")
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            is_function_calling_model=True,
            is_chat_model=True,
        )
    elif api_key := os.environ.get("TOGETHER_AI_API_KEY", None):
        logger.info("Using Together AI API.")
        from llama_index.llms.openai_like import OpenAILike

        Settings.llm = OpenAILike(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            api_base="https://api.together.xyz/v1",
            api_key=api_key,
            is_function_calling_model=True,
            is_chat_model=True,
        )
    else:
        logger.info("Using Ollama's OpenAI-compatible API.")
        from llama_index.llms.openai_like import OpenAILike

        Settings.llm = OpenAILike(
            model="llama3.1",
            api_base="http://localhost:11434/v1",
            # api_base="http://10.147.20.237:11434/v1",
            api_key="ollama",
            is_function_calling_model=True,
            is_chat_model=True,
        )

    Settings.embed_model = OllamaEmbedding(
        # https://ollama.com/library/nomic-embed-text
        model_name="nomic-embed-text",
        # Uncomment the following line to use the LLM server running on my gaming PC.
        # base_url="http://10.147.20.237:11434",
    )


set_up_llama_index()


def init() -> Tuple[ChatMemoryBuffer, Dict[str, AgentRunner], AgentRunner]:
    chat_memory = ChatMemoryBuffer.from_defaults()
    participants: Dict[str, AgentRunner] = {
        "Alice": OpenAIAgent.from_tools(
            tools=[],
            system_prompt="Your name is Alice. You are casually texting with a group of friends. You are kind and helpful.",
            memory=chat_memory,
        ),
        "Bob": OpenAIAgent.from_tools(
            tools=[],
            system_prompt="Your name is Bob. You are casually texting with a group of friends. You are aggressive but caring.",
            memory=chat_memory,
        ),
    }
    judge: AgentRunner = OpenAIAgent.from_tools(
        tools=[],
        system_prompt="You are the judge. You are impartial and fair. "
        "You are here to help your friends resolve their disputes. "
        "Given a series of chat messages, determine whether they have reached an agreement or a truce. "
        'If they did, say "yes". '
        'If things has really escalated, say "stop". '
        'Otherwise, say "keep going". '
        "Use literally these words. Do not even change the capitalization or add punctuation.",
        memory=chat_memory,
    )
    return chat_memory, participants, judge


@cl.on_chat_start
async def on_chat_start():
    chat_history, participants, judge = init()
    cl.user_session.set(
        "chat_history",
        chat_history,
    )
    cl.user_session.set(
        "participants",
        participants,
    )
    cl.user_session.set(
        "judge",
        judge,
    )


@cl.on_message
async def on_message(message: cl.Message):
    """
    This weird combination of `asyncio.create_task(cl.make_async` makes `handle_inquiry` non-blocking. How cool is that?
    """
    asynchronously_handle_inquiry: Callable = cl.make_async(handle_inquiry)
    coroutine = asynchronously_handle_inquiry(
        user_input=message.content,
        chat_memory=cl.user_session.get("chat_history"),
        participants=cl.user_session.get("participants"),
        judge=cl.user_session.get("judge"),
        should_use_chainlit=True,
    )
    asyncio.create_task(coroutine)


def handle_inquiry(
    user_input: str,
    chat_memory: ChatMemoryBuffer,
    participants: Dict[str, AgentRunner],
    judge: AgentRunner,
    should_use_chainlit: bool = False,
    should_judgment_be_visible: bool = False,
):
    should_keep_going = True
    round_id = 0
    chat_memory.put(
        ChatMessage(
            content=f"(speaker: {user_name}) {user_input}", author=MessageRole.USER
        )
    )
    if shared_states["is_conversation_going"]:
        print(">>>>>>>>>> The conversation is already going on <<<<<<<<<<<<")
        return
    shared_states["is_conversation_going"] = True
    while should_keep_going:
        round_id += 1
        print(
            f"=============================== Round {round_id} ==============================="
        )
        for name, participant in participants.items():
            prefix = f"(speaker: {name})"
            change_point_of_view(chat_memory, prefix)
            # The last message will never be from this participant,
            # so we can safely pop it & use it to kick off the `chat` method.
            last_message = pop_last_message(chat_memory)
            response = participant.chat(
                # The last message will always begin with "(speaker: {name})", so we can simply access the content.
                last_message.content
            ).response
            if should_use_chainlit:
                message = cl.Message(
                    content=response,
                    author=name,
                )
                cl.run_sync(message.send())
            print(
                f"-------------------------------- {name} says -------------------------------- \n{response}"
            )
            # This LLM response is also appended to the chat memory. Let's temper it a bit.
            last_message = pop_last_message(chat_memory)
            if not last_message.content.startswith(prefix):
                last_message.content = f"{prefix} {last_message.content}"
            chat_memory.chat_store.add_message(chat_memory.chat_store_key, last_message)
        change_point_of_view(
            chat_memory,
            # No message will be from the judge, so this method call will effectively mark all messages as "user messages".
            "(speaker: judge)",
        )
        judgment: str = judge.chat(
            "Have they reached an agreement or not? [yes/keep going/stop]"
        ).response
        print(
            f"-------------------------------- The judge says -------------------------------- \n{judgment}"
        )
        if judgment == "yes":
            print("The judge has decided that you have reached an agreement.")
            should_keep_going = False
        elif judgment == "stop":
            print("The judge has decided that things have really escalated.")
            should_keep_going = False
        else:
            print("The judge has decided that you should keep going.")
            should_keep_going = True
        if should_use_chainlit and should_judgment_be_visible:
            message = cl.Message(
                content=judgment,
                author="judge",
            )
            cl.run_sync(message.send())
        # Judge's response should not be heard by the participants, so does the prompt to the judge.
        pop_last_message(chat_memory)
        pop_last_message(chat_memory)
    shared_states["is_conversation_going"] = False


def change_point_of_view(chat_memory: ChatMemoryBuffer, prefix: str):
    """
    Change the point of view of the last message in the chat memory.

    Caveat: This relies on the fact that the ChatMemoryBuffer:
    1. retains `prefix` literally for all messages, and
    2. provides a read-write view of the chat history, because we'll be editing the messages in-place.
    """
    all_messages: List[ChatMessage] = chat_memory.chat_store.get_messages(
        chat_memory.chat_store_key
    )
    for message in all_messages:
        if message.role not in (MessageRole.ASSISTANT, MessageRole.USER):
            continue
        if message.content.startswith(prefix):
            # "That's me!"
            message.role = MessageRole.ASSISTANT
        else:
            message.role = MessageRole.USER


def pop_last_message(chat_memory):
    """
    Pop the last message from the chat memory.
    """
    all_messages = chat_memory.chat_store.get_messages(chat_memory.chat_store_key)
    last_message = all_messages[-1]
    chat_memory.chat_store.delete_last_message(chat_memory.chat_store_key)
    return last_message


if __name__ == "__main__":
    # If Python’s builtin readline module is previously loaded, elaborate line editing and history features will be available.

    # https://rich.readthedocs.io/en/stable/console.html#input
    from rich.console import Console

    console = Console()
    chat_history, participants, judge = init()
    user_input = "Yo srsly should I vote for Trump?"
    handle_inquiry(
        user_input=user_input,
        chat_memory=chat_history,
        participants=participants,
        judge=judge,
        should_use_chainlit=False,
    )
