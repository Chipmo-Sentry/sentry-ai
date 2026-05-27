"""Jinja-based prompt template loader."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(default=False),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_prompt(template_name: str, **context: object) -> str:
    """Render a Jinja template from prompts/ with the given context."""
    template = _env.get_template(template_name)
    return template.render(**context)
