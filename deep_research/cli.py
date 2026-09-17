from __future__ import annotations

import argparse
from pathlib import Path

from .local_ai import discover_model_files, discover_server_binaries, start_local_server
from .project import ResearchProject, create_project


def _safe_text(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).replace("\ufeff", "").replace("\x00", "")


def _handle_init(args: argparse.Namespace) -> int:
    root = Path(args.project).expanduser().resolve()
    project = create_project(root)
    print(f"Created project at {project.root}")
    return 0


def _handle_add_source(args: argparse.Namespace) -> int:
    project = ResearchProject(args.project)
    project.create()
    result = project.add_source(args.source_path, title=args.title, url=args.url)
    print(f"Added source: {result['title']} ({result['source_id']})")
    print(f"Claims extracted: {len(project.list_claims(result['document_id']))}")
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    project = ResearchProject(args.project)
    project.create()
    claims = project.analyze_all()
    print(f"Extracted {len(claims)} claims across the project.")
    return 0


def _handle_search(args: argparse.Namespace) -> int:
    project = ResearchProject(args.project)
    project.create()
    hits = project.search(args.query, limit=args.limit)
    if not hits:
        print("No matching evidence found.")
        return 0
    for hit in hits:
        text = _safe_text(hit.get("text"))
        location = _safe_text(hit.get("location"))
        print(f"- {text} | type={_safe_text(hit.get('claim_type'))} | location={location}")
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    project = ResearchProject(args.project)
    project.create()
    markdown = project.generate_report(title=args.title)
    output_path = project.exports_dir / "final_report.md"
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Report written to {output_path}")
    return 0


def _handle_local_ai(args: argparse.Namespace) -> int:
    if args.list_models:
        models = discover_model_files(args.model_roots)
        for model in models:
            print(model)
        return 0
    if args.list_servers:
        servers = discover_server_binaries(args.model_roots)
        for server in servers:
            print(server)
        return 0

    try:
        details = start_local_server(model_path=args.model, port=args.port, server_binary=args.server_binary)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print("Local model server started successfully.")
    print(f"Model: {details['model_path']}")
    print(f"Health: {details['health_url']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Accuracy-first local research system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a new research project directory")
    init_parser.add_argument("project", help="Project directory path")
    init_parser.set_defaults(func=_handle_init)

    add_source = subparsers.add_parser("add-source", help="Ingest a source file into the project")
    add_source.add_argument("project", help="Project directory path")
    add_source.add_argument("source_path", help="Source file to ingest")
    add_source.add_argument("--title", default=None, help="Custom title for the source")
    add_source.add_argument("--url", default=None, help="Optional source URL")
    add_source.set_defaults(func=_handle_add_source)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze all project documents")
    analyze_parser.add_argument("project", help="Project directory path")
    analyze_parser.set_defaults(func=_handle_analyze)

    search_parser = subparsers.add_parser("search", help="Search claim evidence")
    search_parser.add_argument("project", help="Project directory path")
    search_parser.add_argument("query", help="Search phrase")
    search_parser.add_argument("--limit", type=int, default=10)
    search_parser.set_defaults(func=_handle_search)

    report_parser = subparsers.add_parser("report", help="Generate a citation-grounded research report")
    report_parser.add_argument("project", help="Project directory path")
    report_parser.add_argument("--title", default="Local research report")
    report_parser.set_defaults(func=_handle_report)

    local_ai_parser = subparsers.add_parser("local-ai", help="Discover and start a local GGUF-backed model server")
    local_ai_parser.add_argument("--model", help="Path to a .gguf model file")
    local_ai_parser.add_argument("--server-binary", help="Path to llama-server.exe or llama-server")
    local_ai_parser.add_argument("--port", type=int, default=8080)
    local_ai_parser.add_argument("--list-models", action="store_true", help="List discovered GGUF models and exit")
    local_ai_parser.add_argument("--list-servers", action="store_true", help="List discovered llama-server binaries and exit")
    local_ai_parser.add_argument("--model-root", dest="model_roots", action="append", default=[], help="Directory to scan for local models or servers")
    local_ai_parser.set_defaults(func=_handle_local_ai)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
