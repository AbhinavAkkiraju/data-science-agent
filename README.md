# Data Science Agent

This repository contains two LLM-guided optimization examples built with Trace/OptoPrime:

- `house-prices/`: regression workflow for the Kaggle House Prices task
- `spaceship-titanic/`: classification workflow for the Kaggle Spaceship Titanic task

## Prerequisites

- Python `>=3.10,<3.13` recommended
- A Trace-compatible LLM endpoint and API key

## Setup

### 1. Create and activate a virtual environment

Install `uv` first if you do not already have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create and activate the environment:

```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 3. Configure your LLM credentials

Copy the example file and fill in your real values:

```bash
cp .env.example .env
```

The training scripts load `.env` automatically at startup, so you do not need to export these values manually unless you prefer shell environment variables.

Trace supports multiple backends. In the Trace implementation, `LiteLLM` is the general provider-backed path and `CustomLLM` is for an OpenAI-compatible server you host or proxy yourself. See the upstream Trace backend logic here: [Trace `llm.py`](https://raw.githubusercontent.com/microsoft/Trace/main/opto/utils/llm.py).

#### Option A: OpenAI

Use Trace's `LiteLLM` backend with your OpenAI API key:

```dotenv
TRACE_DEFAULT_LLM_BACKEND=LiteLLM
TRACE_LITELLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_openai_api_key_here
```

#### Option B: Anthropic

Also use `LiteLLM`, but provide an Anthropic model plus your Anthropic key:

```dotenv
TRACE_DEFAULT_LLM_BACKEND=LiteLLM
TRACE_LITELLM_MODEL=anthropic/claude-3-5-sonnet-latest
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

#### Option C: OpenRouter

For OpenRouter, keep the `LiteLLM` backend and use an OpenRouter model name plus your OpenRouter key:

```dotenv
TRACE_DEFAULT_LLM_BACKEND=LiteLLM
TRACE_LITELLM_MODEL=openrouter/openai/gpt-4o-mini
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

#### Option D: Your own OpenAI-compatible server

If you run your own gateway, proxy, or model server, use Trace's `CustomLLM` backend instead. This is the right setup for a LiteLLM proxy or any OpenAI-compatible endpoint:

```dotenv
TRACE_DEFAULT_LLM_BACKEND=CustomLLM
TRACE_CUSTOMLLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
TRACE_CUSTOMLLM_URL=http://your-trace-gateway-url/
TRACE_CUSTOMLLM_API_KEY=your_api_key_here
```

In practice:

- Use `LiteLLM` for direct provider access such as OpenAI, Anthropic, or OpenRouter.
- Use `CustomLLM` when you have a single custom endpoint that already speaks the OpenAI-compatible chat completions API.

## Datasets

Download the datasets from Kaggle:

- Spaceship Titanic: [Kaggle competition data](https://www.kaggle.com/competitions/spaceship-titanic/data)
- House Prices: [Kaggle competition overview](https://www.kaggle.com/competitions/home-data-for-ml-course/overview)

After downloading, place the CSV files in these locations:

- `house-prices/data/train.csv`
- `house-prices/data/test.csv`
- `spaceship-titanic/data/train.csv`
- `spaceship-titanic/data/test.csv`

## Running the code

Run commands from the repository root.

### House Prices

Single-function version:

```bash
python house-prices/one_function.py
```

Multi-function version:

```bash
python house-prices/multi_function.py
```

### Spaceship Titanic

Single-function version:

```bash
python spaceship-titanic/one_function.py
```

Multi-function version:

```bash
python spaceship-titanic/multi_function.py
```

## Outputs

Depending on the script, runs write artifacts such as:

- `submission.csv`
- `agent.pkl`
- performance plots like `*.png` or `*.pdf`

These files are currently written to the working directory where the script is launched.

## Notes

- If `.env` is missing the variables required for the backend you selected, the scripts will stop early with a clear error.
- The default `.env.example` now favors `LiteLLM` so people can plug in standard provider keys more easily.
