from pathlib import Path

from deep_research.project import ResearchProject


def test_project_workflow(tmp_path: Path) -> None:
    project_root = tmp_path / "demo_project"
    project = ResearchProject(project_root)
    project.create()

    source_file = tmp_path / "source.txt"
    source_file.write_text(
        "# Clinical summary\n\n"
        "The intervention reduced the risk of adverse events by 18 percent in the trial.\n\n"
        "The analysis found a statistically significant difference after twelve weeks.\n",
        encoding="utf-8",
    )

    result = project.add_source(source_file, title="Trial summary")
    assert result["source_id"]
    assert project.list_sources()

    quality = project.record_source_quality(
        result["source_id"],
        score=0.91,
        peer_reviewed=True,
        primary_source=True,
        publication_date="2026-01-15",
        methodology_transparency=0.82,
        conflict_disclosure=True,
        independence_notes="Independent trial report with transparent methods.",
    )
    assert quality["source_id"] == result["source_id"]
    assert quality["score"] == 0.91

    hits = project.search("adverse events")
    assert hits

    contradictions = project.evaluate_contradictions(project.list_claims(result["document_id"]))
    assert isinstance(contradictions, list)

    index_rows = project.build_retrieval_index(result["document_id"])
    assert index_rows

    analysis = project.analyze_document(result["document_id"])
    assert analysis["verification_status"] == "verified"

    benchmark = project.benchmark_eval("What is the treatment effect?", ["Trial summary"], unsupported_claim_rate=0.03, citation_accuracy=0.9, contradiction_detection=0.8)
    assert benchmark["citation_accuracy"] == 0.9

    question = project.define_research_question(
        "What is the treatment effect?",
        primary_question="What is the treatment effect on adverse events?",
        subquestions=["What was measured?", "Was there a control group?"],
        date_range="2025-2026",
        geographic_scope="US",
        population="Adults",
        definitions=["adverse events", "treatment effect"],
        inclusion_criteria=["adult trial participants"],
        exclusion_criteria=["non-clinical observational studies"],
        source_types=["trial report", "systematic review"],
    )
    assert question["primary_question"] == "What is the treatment effect on adverse events?"

    plan = project.build_source_plan(question["id"], ["adverse events", "treatment effect"], ["trial report"], "hybrid retrieval")
    assert plan["strategy"] == "hybrid retrieval"

    gap = project.record_gap(question["id"], "Only one source was reviewed.", severity="high", recommendation="Add more independent sources.")
    assert gap["severity"] == "high"

    report = project.generate_report(title="Demo report")
    assert "Executive summary" in report
    assert "18 percent" in report
    assert "Trial summary" in report
    assert "quality=0.91" in report
    assert "Contradiction analysis" in report
    assert "Verification pass" in report
    assert "Retrieval index" in report
    assert "Document analyses" in report
    assert "Research questions and plans" in report
    assert "Evidence gaps" in report
