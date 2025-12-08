import re
from typing import Dict, Any, Optional

def _extract_code_from_block(block: str) -> str:
    """
    Take a string that might contain a ```python ... ``` fenced block
    and return just the Python code inside.
    """
    block = block.strip()

    fence_match = re.match(r"^```(?:python)?\s*(.*?)```$", block, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()

    return block


def run_python_block(block: str, env: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a Python code block in a sandboxed local environment and
    return the resulting locals() dict.

    This is used by the teacher orchestrator to verify numerical parts
    of the solution (e.g., recompute Δf, k, etc.).
    """
    code = _extract_code_from_block(block)

    local_env: Dict[str, Any] = {}
    if env:
        local_env.update(env)

    # NOTE: this assumes you trust the code being run (it comes from your generator),
    # so it's okay in your controlled data-gen environment.
    exec(code, {}, local_env)
    return local_env
