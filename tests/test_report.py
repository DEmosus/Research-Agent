"""Tests for report formatting (final_answer + build_report). Pure functions, no mocking needed."""

from src.report import build_report, final_answer


def test_final_answer_joins_list_fields_as_bullets():
    result = final_answer.invoke({
        "introduction": "Intro.",
        "research_steps": ["Step one.", "Step two."],
        "main_body": "Body.",
        "conclusion": "Conclusion.",
        "sources": ["Source A", "Source B"],
    })
    assert "- Step one." in result
    assert "- Step two." in result
    assert "- Source A" in result
    assert "Intro." in result


def test_final_answer_accepts_plain_string_fields():
    result = final_answer.invoke({
        "introduction": "Intro.",
        "research_steps": "Just one paragraph of steps.",
        "main_body": "Body.",
        "conclusion": "Conclusion.",
        "sources": "Just one source line.",
    })
    assert "Just one paragraph of steps." in result
    assert "Just one source line." in result


def test_build_report_renders_all_sections():
    output = {
        "introduction": "Intro text.",
        "research_steps": ["Searched X.", "Searched Y."],
        "main_body": "The main body of the report.",
        "conclusion": "Final thoughts.",
        "sources": ["ArXiv 1234.5678", "example.com"],
    }
    report = build_report(output)
    assert "INTRODUCTION" in report
    assert "Intro text." in report
    assert "RESEARCH STEPS" in report
    assert "- Searched X." in report
    assert "REPORT" in report
    assert "The main body of the report." in report
    assert "CONCLUSION" in report
    assert "SOURCES" in report
    assert "- ArXiv 1234.5678" in report
