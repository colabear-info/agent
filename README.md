# <img width="50" alt="logo" src="public/logo_light.png"> Besties: AI group chat

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fcolabear-info%2Fagent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

## Technical overview

In a group chat with humans, it's a common scenario where you raise a topic and people start discussing it, during which you can interrupt / contribute to the conversation at any time.

This is not trivial with LLMs, because:
- "Chat completion" models are usually trained on datasets with **only two speakers** ("user"/"human" and "assistant"/"AI").
- Chatbot UI usually disables the input field while the AI is responding, which prevents humans from interrupting the AI participants.

This project demonstrates how you can solve these problems by:
- **manipulating the chat history** before each LLM inference to inform the model about different speakers.
- looping through AI participants **in a separate thread**, which frees up the main thread for user input.

## Usage

### Pre-requisites

There are a couple of things you have to do manually before you can start using the chatbot.

1. Clone the repository ([how](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)).
2. **Install the required binary, standalone programs**. These are not Python packages, so they aren't managed by `pyproject.toml`.
3. **Self-serve a text embedding model**. This model "translates" your text into numbers, so that the computer can understand you.
4. **Choose a way to serve a large language model (LLM)**. You can either use OpenAI's API or self-host a local LLM with Ollama.

No need to explicitly install Python packages. `uv`, the package manager of our choice, will implicitly install the required packages when you boot up the chatbot for the first time.

#### Install the required binary programs

These are the binary programs that you need to have ready before running besties:
- Written in Python, this project uses the Rust-based package manager [`uv`](https://docs.astral.sh/uv/). It does not require you to explicitly create a virtual environment.
- As aforementioned, if you decide to self-host a LLM, install Ollama.

If you are on macOS, you can install these programs using Homebrew:

```shell
brew install uv ollama
```

#### Self-serve an embedding model

Ensure that you have a local Ollama server running:

```shell
ollama serve
```

and then:

```shell
ollama pull nomic-embed-text
```

#### Bring your own large language model (LLM)

The easiest (and perhaps highest-quality) way would be to provide an API key to OpenAI. Simply add `OPENAI_API_KEY=sk-...` to a `.env` file in the project root.

With the absence of an OpenAI API key, the chatbot will default to using [Ollama](https://ollama.com/download), a program that serves LLMs locally.
Ensure that your local Ollama server has already downloaded the `llama3.1` model. If you haven't (or aren't sure), run `ollama pull llama3.1`.

### Running the Chatbot

Create a separate terminal for each command:
1. Start serving **Ollama** (for locally inferencing embedding & language models) by running `ollama serve`. It should be listening at `http://localhost:11434/v1`.
2. Start serving **Phoenix** (for debugging thought chains) by running `uv run phoenix serve`.
3. Finally, start serving the **chatbot** by running `uv run chainlit run main.py -w`.

## Troubleshooting

If you see:

```
  File ".../llvmlite-0.43.0.tar.gz/ffi/build.py", line 142, in main_posix
    raise RuntimeError(msg) from None
RuntimeError: Could not find a `llvm-config` binary. There are a number of reasons this could occur, please see: https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#using-pip for help.
error: command '.../bin/python' failed with exit code 1
```

Then run:

```shell
brew install llvm
```

If your `uv run phoenix serve` command fails with:

```
Traceback (most recent call last):
  File "besties/.venv/bin/phoenix", line 5, in <module>
    from phoenix.server.main import main
  File "besties/.venv/lib/python3.11/site-packages/phoenix/__init__.py", line 12, in <module>
    from .session.session import (
  File ".venv/lib/python3.11/site-packages/phoenix/session/session.py", line 41, in <module>
    from phoenix.core.model_schema_adapter import create_model_from_inferences
  File ".venv/lib/python3.11/site-packages/phoenix/core/model_schema_adapter.py", line 11, in <module>
    from phoenix.core.model_schema import Embedding, Model, RetrievalEmbedding, Schema
  File ".venv/lib/python3.11/site-packages/phoenix/core/model_schema.py", line 554, in <module>
    class ModelData(ObjectProxy, ABC):  # type: ignore
TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
```

then you can work around the problem for now by [serving Arize Phoenix from a Docker container](https://docs.arize.com/phoenix/deployment/docker):

```shell
docker run -p 6006:6006 -p 4317:4317 -i -t arizephoenix/phoenix:latest
```
