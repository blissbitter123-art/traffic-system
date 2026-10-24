"""Tests for executive briefing and reporting generation."""

from pathlib import Path
from src.reporting import generate_executive_briefing, export_markdown_report


def test_generate_executive_briefing_schema():
    briefing = generate_executive_briefing()
    assert isinstance(briefing, dict)
    assert "title" in briefing
    assert "dataset_summary" in briefing
    assert "triage_summary" in briefing
    assert "top_priority_corridors" in briefing
    assert "strategic_recommendations" in briefing
    assert len(briefing["strategic_recommendations"]) >= 3


def test_export_markdown_report(tmp_path: Path):
    target = tmp_path / "test_report.md"
    result = export_markdown_report(target)
    assert result.exists()
    content = result.read_text(encoding="utf-8")
    assert "Municipal Traffic Safety Intelligence Briefing" in content
    assert "Executive Summary & Triage Overview" in content
