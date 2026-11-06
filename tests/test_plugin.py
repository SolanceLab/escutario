# Escutário — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0, see LICENSE.md. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The plugin's two manifests, its skill and the changelog name the same version, and the skill names the real commands."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def version():
    return re.search(r"^## \[?(\d+\.\d+\.\d+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1)


def test_both_manifests_carry_the_room_name_and_changelog_version():
    for rel in ("plugin/.claude-plugin/plugin.json", "plugin/.codex-plugin/plugin.json"):
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert data["name"] == "escutario"
        assert data["version"] == version()


def test_codex_manifest_is_skill_only():
    data = json.loads((ROOT / "plugin/.codex-plugin/plugin.json").read_text(encoding="utf-8"))
    assert data["skills"] == "./skills/"
    assert "mcpServers" not in data


def test_skill_pins_its_release_and_names_real_modules():
    skill = (ROOT / "plugin/skills/escutario/SKILL.md").read_text(encoding="utf-8")
    assert "--branch v%s" % version() in skill
    for module in ("escutario.listen", "escutario.view", "escutario.words"):
        assert module in skill
        assert (ROOT / (module.replace(".", "/") + ".py")).is_file()
