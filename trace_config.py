import os
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parent
ENV_FILE = REPO_ROOT / ".env"


def configure_trace_environment() -> None:
    load_dotenv(ENV_FILE)
    load_dotenv()
    backend = os.environ.setdefault("TRACE_DEFAULT_LLM_BACKEND", "LiteLLM")

    if backend == "CustomLLM":
        required_env_vars = (
            "TRACE_CUSTOMLLM_MODEL",
            "TRACE_CUSTOMLLM_URL",
            "TRACE_CUSTOMLLM_API_KEY",
        )
    elif backend == "LiteLLM":
        required_env_vars = ("TRACE_LITELLM_MODEL",)
    else:
        required_env_vars = ()

    missing_env_vars = [name for name in required_env_vars if not os.getenv(name)]
    if missing_env_vars:
        missing_str = ", ".join(missing_env_vars)
        raise RuntimeError(
            f"Missing required Trace environment variables for backend {backend}: {missing_str}. "
            f"Create {ENV_FILE} from .env.example or export them in your shell."
        )
