# Deep-Research

A practical implementation of an accuracy-first local research system inspired by the design brief in the attached document.

The project gives you a working local workflow for:

- ingesting source files and preserving provenance
- storing raw artifacts and parsed text on disk
- splitting documents into section-level evidence units
- extracting claim-level findings from source text
- searching the claim ledger for relevant evidence
- generating a citation-grounded markdown report
- discovering and launching a local GGUF model stack similar to the portable F: setup

## Architecture

The system stores metadata in SQLite, keeps raw source files under `raw/`, and creates report artifacts under `exports/`. Each source is copied into the project with a content hash so the original can be preserved while analysis is performed on a durable local copy.

It also includes a lightweight local-model launcher in `deep_research/local_ai.py` that can:

- scan nearby directories for `.gguf` model files
- scan for `llama-server` binaries
- start a local inference server on `127.0.0.1:<port>`
- expose a simple command-line workflow similar to a portable local AI stack

## Quick start

```bash
python -m deep_research init demo-project
python -m deep_research add-source demo-project ./sample-source.txt --title "Sample source"
python -m deep_research analyze demo-project
python -m deep_research search demo-project "risk reduction"
python -m deep_research report demo-project --title "Example report"
```

### Local model launcher

```bash
python -m deep_research.local_ai --list-models
python -m deep_research.local_ai --list-servers
python -m deep_research.local_ai --model "C:\path\to\model.gguf" --server-binary "C:\path\to\llama-server.exe"
```

On Windows, you can also run the helper batch file:

```bat
run_research.bat init demo-project
run_research.bat local-ai --list-models
```

## Project layout

```text
demo-project/
  raw/
  extracted/
  analyses/
  exports/
  research.sqlite3
```

## Supported local models and where to get them

This project is intentionally model-agnostic. It does not bind to a specific provider or model family. The local launcher works with any compatible GGUF model you can place on disk, including:

- Llama 3.x GGUF
- Mistral GGUF
- Qwen GGUF
- other OpenAI-compatible or GGUF-converted local models

Common sources include:

- local model repositories you already have installed
- Hugging Face model files converted to GGUF
- portable AI bundles placed on an external drive like `F:\`
- custom local model directories maintained by your own stack

You do not need to install a special package for the repo itself; the launcher only looks for model files and a matching `llama-server` executable on disk.

## What is implemented

This first pass focuses on the core of the design brief: document ingestion, section preservation, evidence extraction, retrieval, and report generation. It also includes a practical local-model bootstrap layer so the project can run alongside an existing portable AI stack rather than being isolated from it.

## Validation

```bash
pytest -q
```
