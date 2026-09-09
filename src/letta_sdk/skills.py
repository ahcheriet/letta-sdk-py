"""Skill seeding for ``create_agent(skills=...)``.

Port of the TypeScript SDK's ``skill-loading.ts`` portable core, with the
directory loader inlined (Python has a real filesystem, so no injected
loader is needed).

A skill's canonical shape is a directory: ``SKILL.md`` plus optional support
files (``scripts/``, ``references/``, ...). The platform stores
``skills/{name}/SKILL.md`` as a memory block labeled ``skills/{name}``; the
block value must be the SKILL.md body (without frontmatter) — the server
synthesizes the frontmatter from the block description.

Support files have no block representation: the app-server backend has no
push path for them (the TS SDK requires the Cloud backend for that), so
skills with support files are rejected rather than seeded incomplete.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "AgentSkill",
    "assert_valid_skill_name",
    "load_skill_directory",
    "parse_skill_markdown",
    "resolve_skill_items",
    "skill_memory_blocks",
    "skills_have_support_files",
]

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


@dataclass(slots=True)
class AgentSkill:
    """A skill to seed into a new agent's memory filesystem at creation.

    The instructions become ``skills/{name}/SKILL.md`` in the agent's memory
    repo (stored as the memory block labeled ``skills/{name}``), so the
    agent owns and can edit the skill like any other memory.
    """

    #: Directory name: lowercase letters, digits, ".", "_", "-".
    name: str
    #: Trigger text: when the agent should load this skill. Rendered as the
    #: SKILL.md frontmatter description and always visible to the agent.
    description: str
    #: SKILL.md body (markdown, without frontmatter).
    instructions: str
    #: Optional support files, keyed by path relative to the skill directory
    #: (e.g. ``"scripts/convert.sh"``). Not deliverable on the app-server
    #: backend — rejected by :func:`resolve_skill_items` callers' push gate.
    files: dict[str, bytes] | None = None


def assert_valid_skill_name(name: str) -> None:
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            f'Invalid skill name "{name}". Skill names are directory names: '
            'lowercase letters, digits, ".", "_", "-" '
            '(e.g. "generating-voice-memos").'
        )


def parse_skill_markdown(content: str) -> tuple[str | None, str | None, str]:
    """Parse SKILL.md into ``(name, description, body)``.

    Port of the TS ``parseSkillMarkdown``: a ``---`` fenced frontmatter of
    simple ``key: value`` lines (with YAML folded/literal scalar support
    and quote stripping). Missing frontmatter means the whole content is
    the body.
    """
    if not content.startswith("---\n") and not content.startswith("---\r\n"):
        return (None, None, content)
    fence = re.search(r"\r?\n---[ \t]*(\r?\n|$)", content[3:])
    if fence is None:
        return (None, None, content)
    yaml_start = 5 if content.startswith("---\r\n") else 4
    yaml_end = 3 + fence.start()
    yaml_text = content[yaml_start:yaml_end]
    body = content[yaml_end + len(fence.group(0)) :]

    fields: dict[str, str] = {}
    lines = re.split(r"\r?\n", yaml_text)
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line or line[0].isspace():
            i += 1
            continue
        colon = line.find(":")
        if colon <= 0:
            i += 1
            continue
        key = line[:colon].strip()
        value = line[colon + 1 :].strip()
        if value in (">", ">-", "|", "|-"):
            folded: list[str] = []
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            while nxt is not None and (nxt == "" or nxt[0].isspace()):
                i += 1
                folded.append(nxt.strip())
                nxt = lines[i + 1] if i + 1 < len(lines) else None
            value = " ".join(part for part in folded if part)
        elif len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]
        fields[key] = value
        i += 1

    return (fields.get("name"), fields.get("description"), body)


def load_skill_directory(dir_path: str | os.PathLike[str]) -> AgentSkill:
    """Load a skill directory (``SKILL.md`` + support files) from disk."""
    path = Path(dir_path)
    if not path.is_dir():
        raise ValueError(f"Skill path is not a directory: {dir_path}")
    skill_md = path / "SKILL.md"
    if not skill_md.is_file():
        raise ValueError(f"Skill directory has no SKILL.md: {dir_path}")

    content = skill_md.read_text(encoding="utf-8")
    name, description, body = parse_skill_markdown(content)
    if name is None:
        name = path.name
    assert_valid_skill_name(name)
    if not description:
        raise ValueError(
            f"SKILL.md in {dir_path} has no frontmatter description. "
            "The description is the skill's trigger text; it is required."
        )

    # Collect support files (everything except SKILL.md), preserving layout.
    files: dict[str, bytes] = {}
    for entry in sorted(path.rglob("*")):
        if not entry.is_file():
            continue
        rel = entry.relative_to(path).as_posix()
        if rel == "SKILL.md":
            continue
        files[rel] = entry.read_bytes()

    return AgentSkill(
        name=name,
        description=description,
        instructions=body,
        files=files or None,
    )


def _validate_inline_skill(skill: AgentSkill) -> None:
    assert_valid_skill_name(skill.name)
    if not skill.instructions or not skill.instructions.strip():
        raise ValueError(f'Skill "{skill.name}" has empty instructions.')
    if not skill.description or not skill.description.strip():
        raise ValueError(
            f'Skill "{skill.name}" has no description. '
            "The description is the skill's trigger text; it is required."
        )


def _skill_from_dict(item: dict[str, Any]) -> AgentSkill:
    try:
        name = item["name"]
        description = item["description"]
        instructions = item["instructions"]
    except KeyError as exc:
        raise ValueError(
            "Inline skill dicts require 'name', 'description', and "
            f"'instructions' keys (missing {exc})."
        ) from None
    files = item.get("files")
    if files is not None:
        if not isinstance(files, dict) or not all(
            isinstance(k, str) and isinstance(v, (bytes, bytearray))
            for k, v in files.items()
        ):
            raise ValueError(
                "Inline skill 'files' must map relative paths to bytes."
            )
        files = {k: bytes(v) for k, v in files.items()}
    return AgentSkill(
        name=name, description=description, instructions=instructions, files=files
    )


def resolve_skill_items(
    items: list[str | AgentSkill | dict[str, Any]] | None,
) -> list[AgentSkill]:
    """Resolve every skill item to an :class:`AgentSkill`.

    Items may be a path to a skill directory (loaded from disk) or an
    inline :class:`AgentSkill` / dict. Raises on invalid names, empty
    instructions/descriptions, and duplicate names.
    """
    if not items:
        return []
    resolved: list[AgentSkill] = []
    for item in items:
        if isinstance(item, str):
            resolved.append(load_skill_directory(item))
        elif isinstance(item, AgentSkill):
            _validate_inline_skill(item)
            resolved.append(item)
        elif isinstance(item, dict):
            skill = _skill_from_dict(item)
            _validate_inline_skill(skill)
            resolved.append(skill)
        else:
            raise ValueError(
                f"Invalid skill item: {item!r}. Expected a skill directory "
                "path or an AgentSkill."
            )
    seen: set[str] = set()
    for skill in resolved:
        if skill.name in seen:
            raise ValueError(f'Duplicate skill name: "{skill.name}".')
        seen.add(skill.name)
    return resolved


def skills_have_support_files(skills: list[AgentSkill]) -> bool:
    return any(skill.files for skill in skills)


def skill_memory_blocks(skills: list[AgentSkill]) -> list[dict[str, Any]]:
    """Memory blocks that seed the skills: ``skills/{name}`` per skill.

    The value is the SKILL.md body only; the server synthesizes the
    frontmatter from the block description (embedding it in the value
    would double it).
    """
    return [
        {
            "label": f"skills/{skill.name}",
            "value": skill.instructions,
            "description": skill.description,
        }
        for skill in skills
    ]
