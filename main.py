#!/usr/bin/env python
import logging
import os
from typing import Dict, Tuple

import chainlit as cl
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
        "Given a list of opinions, determine whether they have reached an agreement or not. "
        'If they did, say "yes". '
        'If things has really escalated, say "stop". '
        'Otherwise, say "keep going". '
        "Use literally these words. Do not even change the capitalization or add punctuation.",
        memory=chat_memory,
    )
    return chat_memory, participants, judge


@cl.on_chat_start
async def on_chat_start():
    _, participants, judge = init()
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
    ChainLit provides a web GUI for this application.
    """
    handle_inquiry(
        user_input=message.content,
        participants=cl.user_session.get("participants"),
        judge=cl.user_session.get("judge"),
        should_use_chainlit=True,
    )


def handle_inquiry(
    user_input: str,
    participants: Dict[str, AgentRunner],
    judge: AgentRunner,
    should_use_chainlit: bool = False,
    should_judgment_be_visible: bool = False,
):
    should_keep_going = True
    round_id = 0
    # Memory is shared among all agents, so we can just grab the memory of any agent.
    chat_memory: ChatMemoryBuffer = judge.memory
    chat_memory.put(ChatMessage(content=f"{user_input}", author=MessageRole.USER))
    while should_keep_going:
        round_id += 1
        print(
            f"=============================== Round {round_id} ==============================="
        )
        for name, participant in participants.items():
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
            prefix = f"(speaker: {name})"
            if not last_message.content.startswith(prefix):
                last_message.content = f"{prefix} {last_message.content}"
            last_message.role = MessageRole.USER
            chat_memory.chat_store.add_message(chat_memory.chat_store_key, last_message)
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
    _, participants, judge = init()
    user_input = "Yo srsly should I vote for Trump?"
    handle_inquiry(
        user_input=user_input,
        participants=participants,
        judge=judge,
        should_use_chainlit=False,
    )
