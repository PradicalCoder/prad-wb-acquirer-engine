"""The LLM agents. Each agent's prompts live in agents/prompts/ as plain text files."""

from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"


def load_prompt(name: str, **values: object) -> tuple[str, str]:
    """Read agents/prompts/<name>.md and return (version, text).

    The first line is "version: <id>". Bump it whenever the prompt changes: the
    version is part of every cached answer's key, so old answers stop matching.
    Placeholders like {banned_phrases} are filled from `values`.
    """
    header, text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8").split("\n", 1)
    version = header.removeprefix("version:").strip()
    return version, text.strip().format(**values)
