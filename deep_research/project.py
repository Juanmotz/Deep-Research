from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency for PDF support
    PdfReader = None


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self._parts.append(text)

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"p", "li", "div", "section", "article", "h1", "h2", "h3", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "li", "div", "section", "article", "h1", "h2", "h3"}:
            self._parts.append("\n")

    def get_text(self) -> str:
        return "\n".join(self._parts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    mapping = {
        ".txt": "text",
        ".md": "markdown",
        ".htm": "html",
        ".html": "html",
        ".csv": "dataset",
        ".json": "dataset",
        ".pdf": "pdf",
    }
    return mapping.get(suffix, "document")


def _read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        parser = _HTMLTextExtractor()
        parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
        parser.close()
        return parser.get_text().replace("\ufeff", "").replace("\x00", "")
    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("PDF reading requires the optional 'pypdf' dependency.")
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            pages.append((page.extract_text() or "").replace("\ufeff", "").replace("\x00", ""))
        return "\n\n".join(pages)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text.replace("\ufeff", "").replace("\x00", "")


def _split_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = "Document"
    current_lines: list[str] = []

    def flush() -> None:
        block = "\n".join(current_lines).strip()
        if block:
            sections.append((current_heading, block))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue
        if re.match(r"^#{1,6}\s+.+", stripped):
            flush()
            current_heading = re.sub(r"^#+\s*", "", stripped).strip() or "Section"
            current_lines = []
            continue
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .,:;\-]{2,}:", stripped):
            flush()
            current_heading = stripped.rstrip(":").strip() or "Section"
            current_lines = []
            continue
        current_lines.append(stripped)

    flush()
    if not sections:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return [("Document", cleaned)] if cleaned else [("Document", "No extracted text")]
    return sections


def _extract_claims(section_id: str, document_id: str, section_text: str) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", section_text)
    claims: list[dict] = []

    for sentence in sentences:
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if len(cleaned) < 18:
            continue
        lowered = cleaned.lower()
        has_signal = any(
            token in lowered for token in (
                "found",
                "reported",
                "showed",
                "demonstrated",
                "indicates",
                "suggests",
                "reduces",
                "increases",
                "improves",
                "causes",
                "study",
                "result",
                "analysis",
                "conclusion",
            )
        ) or bool(re.search(r"\d", cleaned))
        if not has_signal:
            continue

        claim_type = "quantitative_result" if re.search(r"\d+(?:%|\.\d+%|\.\d+)?", cleaned) else "factual_statement"
        if any(token in lowered for token in ("causes", "reduces", "increases", "improves")):
            claim_type = "causal_claim"
        confidence = min(0.97, max(0.58, 0.62 + (len(cleaned) / 220) * 0.2 + (0.12 if re.search(r"\d", cleaned) else 0.0)))
        claims.append(
            {
                "id": f"claim-{uuid.uuid4().hex[:8]}",
                "document_id": document_id,
                "section_id": section_id,
                "text": cleaned,
                "claim_type": claim_type,
                "confidence": round(confidence, 3),
                "evidence": cleaned,
            }
        )
    return claims


def _normalized_tokens(value: str) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return {token for token in cleaned.split() if len(token) >= 3}


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(overlap) / len(union) if union else 0.0


def _claim_polarity(text: str) -> str:
    lowered = text.lower()
    positive = any(token in lowered for token in ("increase", "improved", "improvement", "stronger", "better", "higher", "growth"))
    negative = any(token in lowered for token in ("reduced", "decrease", "decline", "worse", "lower", "drop", "fell"))
    if positive and not negative:
        return "positive"
    if negative and not positive:
        return "negative"
    return "neutral"


class ResearchProject:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.raw_dir = self.root / "raw"
        self.extracted_dir = self.root / "extracted"
        self.analyses_dir = self.root / "analyses"
        self.exports_dir = self.root / "exports"
        self.db_path = self.root / "research.sqlite3"

    def create(self) -> "ResearchProject":
        self.root.mkdir(parents=True, exist_ok=True)
        for folder in (self.raw_dir, self.extracted_dir, self.analyses_dir, self.exports_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self._init_db()
        return self

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    file_path TEXT,
                    url TEXT,
                    content_hash TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    doc_type TEXT NOT NULL,
                    body TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sections (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    heading TEXT NOT NULL,
                    section_order INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS claims (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    section_id TEXT,
                    text TEXT NOT NULL,
                    claim_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id),
                    FOREIGN KEY(section_id) REFERENCES sections(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    markdown TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_quality (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    peer_reviewed INTEGER NOT NULL DEFAULT 0,
                    primary_source INTEGER NOT NULL DEFAULT 0,
                    publication_date TEXT,
                    methodology_transparency REAL NOT NULL DEFAULT 0,
                    conflict_disclosure INTEGER NOT NULL DEFAULT 0,
                    independence_notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_families (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    family_id TEXT NOT NULL,
                    match_type TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(source_id) REFERENCES sources(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS contradiction_checks (
                    id TEXT PRIMARY KEY,
                    claim_a TEXT NOT NULL,
                    claim_b TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    explanation TEXT,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_runs (
                    id TEXT PRIMARY KEY,
                    report_id TEXT,
                    statement TEXT NOT NULL,
                    supported INTEGER NOT NULL,
                    reasoning TEXT,
                    citation TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(report_id) REFERENCES reports(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_index (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    section_id TEXT,
                    term TEXT NOT NULL,
                    tf REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_analyses (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    inventory TEXT NOT NULL,
                    section_summary TEXT NOT NULL,
                    synthesis TEXT NOT NULL,
                    verification_status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS claim_links (
                    id TEXT PRIMARY KEY,
                    source_claim_id TEXT NOT NULL,
                    target_claim_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmark_runs (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    expected_sources TEXT,
                    unsupported_claim_rate REAL NOT NULL,
                    citation_accuracy REAL NOT NULL,
                    contradiction_detection REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_questions (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    primary_question TEXT,
                    subquestions TEXT,
                    date_range TEXT,
                    geographic_scope TEXT,
                    population TEXT,
                    definitions TEXT,
                    inclusion_criteria TEXT,
                    exclusion_criteria TEXT,
                    source_types TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS source_plans (
                    id TEXT PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    search_terms TEXT NOT NULL,
                    source_types TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES research_questions(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS evidence_gaps (
                    id TEXT PRIMARY KEY,
                    question_id TEXT,
                    gap_text TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    recommendation TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES research_questions(id)
                )
                """
            )
            connection.commit()

    def add_source(self, file_path: str | Path, title: str | None = None, url: str | None = None) -> dict:
        source_path = Path(file_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")

        source_id = f"source-{uuid.uuid4().hex[:8]}"
        content_hash = _sha256(source_path)
        destination = self.raw_dir / f"{source_id}-{source_path.name}"
        shutil.copy2(source_path, destination)

        source_type = _infer_source_type(source_path)
        source_title = title or source_path.stem.replace("_", " ").replace("-", " ").title()
        source_metadata = {
            "path": str(source_path),
            "copied_to": str(destination),
            "hash": content_hash,
            "url": url,
            "source_type": source_type,
        }

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sources (id, title, source_type, file_path, url, content_hash, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, source_title, source_type, str(destination), url, content_hash, json.dumps(source_metadata), _utc_now()),
            )

            document_id = f"doc-{uuid.uuid4().hex[:8]}"
            body = _read_text_file(destination)
            connection.execute(
                "INSERT INTO documents (id, source_id, title, doc_type, body, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (document_id, source_id, source_title, source_type, body, json.dumps({"hash": content_hash}), _utc_now()),
            )

            sections = _split_sections(body)
            for order, (heading, text) in enumerate(sections):
                section_id = f"section-{uuid.uuid4().hex[:8]}"
                connection.execute(
                    "INSERT INTO sections (id, document_id, heading, section_order, text, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (section_id, document_id, heading, order, text, json.dumps({"lines": len(text.splitlines())}), _utc_now()),
                )
                for claim in _extract_claims(section_id, document_id, text):
                    connection.execute(
                        "INSERT INTO claims (id, document_id, section_id, text, claim_type, confidence, evidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            claim["id"],
                            claim["document_id"],
                            claim["section_id"],
                            claim["text"],
                            claim["claim_type"],
                            claim["confidence"],
                            claim["evidence"],
                            _utc_now(),
                        ),
                    )

            connection.commit()

        return {
            "source_id": source_id,
            "document_id": document_id,
            "title": source_title,
            "hash": content_hash,
            "source_type": source_type,
        }

    def list_sources(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, title, source_type, file_path, content_hash, created_at FROM sources ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_claims(self, document_id: str | None = None) -> list[dict]:
        query = "SELECT c.id, c.document_id, c.section_id, c.text, c.claim_type, c.confidence, c.evidence, s.heading AS location FROM claims c LEFT JOIN sections s ON s.id = c.section_id"
        params: list[str] = []
        if document_id:
            query += " WHERE c.document_id = ?"
            params.append(document_id)
        query += " ORDER BY c.confidence DESC, c.created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        phrase = query.strip()
        if not phrase:
            return []
        terms = [term for term in re.findall(r"[A-Za-z0-9]+", phrase.lower()) if len(term) >= 3]
        if not terms:
            return []

        clauses = []
        parameters: list[str] = []
        for term in terms:
            clauses.append("(LOWER(c.text) LIKE ? OR LOWER(s.text) LIKE ?)")
            pattern = f"%{term}%"
            parameters.extend([pattern, pattern])

        sql = "SELECT c.id, c.document_id, c.section_id, c.text, c.claim_type, c.confidence, c.evidence, s.heading AS location FROM claims c LEFT JOIN sections s ON s.id = c.section_id WHERE " + " AND ".join(clauses) + " ORDER BY c.confidence DESC LIMIT ?"
        parameters.append(str(limit))

        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        matches = [dict(row) for row in rows]

        if not matches:
            return []

        scored: list[dict] = []
        for hit in matches:
            text = hit["text"]
            query_tokens = _normalized_tokens(phrase)
            text_tokens = _normalized_tokens(text)
            overlap = len(query_tokens & text_tokens)
            relevance = (overlap / max(1, len(query_tokens))) * 0.6 + (hit["confidence"] or 0.0) * 0.4
            hit["relevance_score"] = round(relevance, 3)
            scoring = _text_similarity(phrase, text)
            hit["similarity_score"] = round(scoring, 3)
            hit["polarity"] = _claim_polarity(text)
            scored.append(hit)
        scored.sort(key=lambda item: (item["relevance_score"], item["confidence"]), reverse=True)
        return scored[:limit]

    def analyze_all(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, title, body FROM documents ORDER BY created_at ASC").fetchall()
            all_claims: list[dict] = []
            for row in rows:
                document_id = row["id"]
                sections = connection.execute("SELECT id, heading, text FROM sections WHERE document_id = ? ORDER BY section_order ASC", (document_id,)).fetchall()
                for section in sections:
                    extracted = _extract_claims(section["id"], document_id, section["text"])
                    if not extracted:
                        continue
                    for claim in extracted:
                        connection.execute(
                            "INSERT INTO claims (id, document_id, section_id, text, claim_type, confidence, evidence, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                claim["id"],
                                claim["document_id"],
                                claim["section_id"],
                                claim["text"],
                                claim["claim_type"],
                                claim["confidence"],
                                claim["evidence"],
                                _utc_now(),
                            ),
                        )
                        all_claims.append({**claim, "location": section["heading"]})
            connection.commit()
        return all_claims

    def record_source_quality(
        self,
        source_id: str,
        *,
        score: float,
        peer_reviewed: bool = False,
        primary_source: bool = False,
        publication_date: str | None = None,
        methodology_transparency: float = 0.0,
        conflict_disclosure: bool = False,
        independence_notes: str | None = None,
    ) -> dict:
        quality_id = f"quality-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO source_quality (id, source_id, score, peer_reviewed, primary_source, publication_date, methodology_transparency, conflict_disclosure, independence_notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    quality_id,
                    source_id,
                    float(score),
                    int(bool(peer_reviewed)),
                    int(bool(primary_source)),
                    publication_date,
                    float(methodology_transparency),
                    int(bool(conflict_disclosure)),
                    independence_notes,
                    _utc_now(),
                ),
            )
            connection.commit()
        return {
            "id": quality_id,
            "source_id": source_id,
            "score": float(score),
            "peer_reviewed": bool(peer_reviewed),
            "primary_source": bool(primary_source),
        }

    def mark_source_family(self, source_id: str, family_id: str, match_type: str, similarity: float) -> dict:
        family_record_id = f"family-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO source_families (id, source_id, family_id, match_type, similarity, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (family_record_id, source_id, family_id, match_type, float(similarity), _utc_now()),
            )
            connection.commit()
        return {
            "id": family_record_id,
            "source_id": source_id,
            "family_id": family_id,
            "match_type": match_type,
            "similarity": float(similarity),
        }

    def evaluate_contradictions(self, claims: list[dict] | None = None) -> list[dict]:
        if claims is None:
            with self._connect() as connection:
                claims = connection.execute(
                    "SELECT id, text, claim_type, confidence FROM claims ORDER BY confidence DESC"
                ).fetchall()
                claims = [dict(row) for row in claims]

        scored: list[dict] = []
        for index, left in enumerate(claims):
            for right in claims[index + 1:]:
                if left["id"] == right["id"]:
                    continue
                left_tokens = _normalized_tokens(left["text"])
                right_tokens = _normalized_tokens(right["text"])
                overlap = len(left_tokens & right_tokens)
                if overlap == 0:
                    continue
                relation = "potential-contradiction"
                explanation = "The claims share some evidence terms but differ in direction or scope."
                if _claim_polarity(left["text"]) != _claim_polarity(right["text"]) and _claim_polarity(left["text"]) != "neutral":
                    relation = "directional-disagreement"
                    explanation = "The claims differ in directional language and should be examined for conflicting evidence."
                score = min(1.0, 0.35 + (overlap / max(1, len(left_tokens | right_tokens))) + abs(float(left.get("confidence", 0.0)) - float(right.get("confidence", 0.0))) * 0.5)
                record = {
                    "id": f"contradiction-{uuid.uuid4().hex[:8]}",
                    "claim_a": left["text"],
                    "claim_b": right["text"],
                    "relation": relation,
                    "explanation": explanation,
                    "score": round(score, 3),
                }
                scored.append(record)
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO contradiction_checks (id, claim_a, claim_b, relation, explanation, score, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (record["id"], record["claim_a"], record["claim_b"], record["relation"], record["explanation"], record["score"], _utc_now()),
                    )
                    connection.commit()
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored

    def verify_claims(self, report_id: str | None = None, statements: list[str] | None = None) -> list[dict]:
        if statements is None:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT text FROM claims ORDER BY confidence DESC"
                ).fetchall()
                statements = [row["text"] for row in rows]

        verification_records: list[dict] = []
        for statement in statements:
            supported = 1 if any(token in statement.lower() for token in ("found", "reported", "showed", "reduced", "increased", "improved")) else 0
            reasoning = "Statement uses explicit evidence-language and appears tied to a source-backed claim." if supported else "Statement is weakly grounded and should be treated as provisional until verified against a source excerpt."
            citation = "Source excerpt retained in claim ledger" if supported else "Missing direct evidence citation"
            record = {
                "id": f"verify-{uuid.uuid4().hex[:8]}",
                "report_id": report_id,
                "statement": statement,
                "supported": supported,
                "reasoning": reasoning,
                "citation": citation,
            }
            verification_records.append(record)
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO verification_runs (id, report_id, statement, supported, reasoning, citation, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record["id"], record["report_id"], record["statement"], record["supported"], record["reasoning"], record["citation"], _utc_now()),
                )
                connection.commit()
        return verification_records

    def build_retrieval_index(self, document_id: str | None = None) -> list[dict]:
        with self._connect() as connection:
            if document_id:
                sections = connection.execute(
                    "SELECT id, heading, text FROM sections WHERE document_id = ? ORDER BY section_order ASC",
                    (document_id,),
                ).fetchall()
            else:
                sections = connection.execute(
                    "SELECT id, heading, text FROM sections ORDER BY section_order ASC"
                ).fetchall()

        index_rows: list[dict] = []
        for section in sections:
            text = section["text"]
            terms = re.findall(r"[A-Za-z0-9]+", text.lower())
            counts: dict[str, int] = {}
            for term in terms:
                if len(term) < 3:
                    continue
                counts[term] = counts.get(term, 0) + 1
            for term, count in counts.items():
                tf = count / max(1, len(terms))
                entry = {
                    "id": f"index-{uuid.uuid4().hex[:8]}",
                    "document_id": document_id or self._connect().execute("SELECT document_id FROM sections WHERE id = ?", (section["id"],)).fetchone()[0],
                    "section_id": section["id"],
                    "term": term,
                    "tf": float(tf),
                }
                index_rows.append(entry)
                with self._connect() as connection:
                    connection.execute(
                        "INSERT INTO retrieval_index (id, document_id, section_id, term, tf, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (entry["id"], entry["document_id"], entry["section_id"], entry["term"], entry["tf"], _utc_now()),
                    )
                    connection.commit()
        return index_rows

    def analyze_document(self, document_id: str) -> dict:
        with self._connect() as connection:
            document = connection.execute("SELECT title, body FROM documents WHERE id = ?", (document_id,)).fetchone()
            if document is None:
                raise ValueError(f"Document not found: {document_id}")
            sections = connection.execute(
                "SELECT id, heading, text FROM sections WHERE document_id = ? ORDER BY section_order ASC",
                (document_id,),
            ).fetchall()

        inventory = f"Document title: {document['title']}; sections: {len(sections)}"
        section_summary = "\n".join(f"{section['heading']}: {section['text'][:220]}" for section in sections)
        synthesis = f"This document covers {len(sections)} sections and emphasizes evidence-grounded findings derived from the supplied text."
        verification_status = "verified" if sections else "unverified"
        analysis_id = f"analysis-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO document_analyses (id, document_id, inventory, section_summary, synthesis, verification_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, document_id, inventory, section_summary, synthesis, verification_status, _utc_now()),
            )
            connection.commit()
        return {
            "id": analysis_id,
            "document_id": document_id,
            "inventory": inventory,
            "section_summary": section_summary,
            "synthesis": synthesis,
            "verification_status": verification_status,
        }

    def link_claims(self, claim_a_id: str, claim_b_id: str, relation: str, similarity: float) -> dict:
        link_id = f"link-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO claim_links (id, source_claim_id, target_claim_id, relation, similarity, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (link_id, claim_a_id, claim_b_id, relation, float(similarity), _utc_now()),
            )
            connection.commit()
        return {
            "id": link_id,
            "source_claim_id": claim_a_id,
            "target_claim_id": claim_b_id,
            "relation": relation,
            "similarity": float(similarity),
        }

    def benchmark_eval(self, question: str, expected_sources: list[str] | None = None, unsupported_claim_rate: float = 0.0, citation_accuracy: float = 0.0, contradiction_detection: float = 0.0) -> dict:
        benchmark_id = f"benchmark-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO benchmark_runs (id, question, expected_sources, unsupported_claim_rate, citation_accuracy, contradiction_detection, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (benchmark_id, question, json.dumps(expected_sources or []), float(unsupported_claim_rate), float(citation_accuracy), float(contradiction_detection), _utc_now()),
            )
            connection.commit()
        return {
            "id": benchmark_id,
            "question": question,
            "expected_sources": expected_sources or [],
            "unsupported_claim_rate": float(unsupported_claim_rate),
            "citation_accuracy": float(citation_accuracy),
            "contradiction_detection": float(contradiction_detection),
        }

    def checkpoint(self, stage: str, status: str, details: str | None = None) -> dict:
        checkpoint_id = f"checkpoint-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO checkpoints (id, stage, status, details, created_at) VALUES (?, ?, ?, ?, ?)",
                (checkpoint_id, stage, status, details, _utc_now()),
            )
            connection.commit()
        return {"id": checkpoint_id, "stage": stage, "status": status, "details": details}

    def define_research_question(
        self,
        question: str,
        *,
        primary_question: str | None = None,
        subquestions: list[str] | None = None,
        date_range: str | None = None,
        geographic_scope: str | None = None,
        population: str | None = None,
        definitions: list[str] | None = None,
        inclusion_criteria: list[str] | None = None,
        exclusion_criteria: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> dict:
        question_id = f"question-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO research_questions (id, question, primary_question, subquestions, date_range, geographic_scope, population, definitions, inclusion_criteria, exclusion_criteria, source_types, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    question,
                    primary_question or question,
                    json.dumps(subquestions or []),
                    date_range,
                    geographic_scope,
                    population,
                    json.dumps(definitions or []),
                    json.dumps(inclusion_criteria or []),
                    json.dumps(exclusion_criteria or []),
                    json.dumps(source_types or []),
                    _utc_now(),
                ),
            )
            connection.commit()
        return {
            "id": question_id,
            "question": question,
            "primary_question": primary_question or question,
            "subquestions": subquestions or [],
            "source_types": source_types or [],
        }

    def build_source_plan(self, question_id: str, search_terms: list[str], source_types: list[str], strategy: str) -> dict:
        plan_id = f"plan-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO source_plans (id, question_id, search_terms, source_types, strategy, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (plan_id, question_id, json.dumps(search_terms), json.dumps(source_types), strategy, _utc_now()),
            )
            connection.commit()
        return {"id": plan_id, "question_id": question_id, "search_terms": search_terms, "source_types": source_types, "strategy": strategy}

    def record_gap(self, question_id: str, gap_text: str, severity: str = "medium", recommendation: str | None = None) -> dict:
        gap_id = f"gap-{uuid.uuid4().hex[:8]}"
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evidence_gaps (id, question_id, gap_text, severity, recommendation, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (gap_id, question_id, gap_text, severity, recommendation, _utc_now()),
            )
            connection.commit()
        return {"id": gap_id, "question_id": question_id, "gap_text": gap_text, "severity": severity, "recommendation": recommendation}

    def generate_report(self, title: str = "Research report") -> str:
        with self._connect() as connection:
            sources = connection.execute("SELECT title, source_type, content_hash, id FROM sources ORDER BY created_at DESC").fetchall()
            claims = connection.execute(
                "SELECT c.text, c.claim_type, c.confidence, s.heading AS location FROM claims c LEFT JOIN sections s ON s.id = c.section_id ORDER BY c.confidence DESC LIMIT 50"
            ).fetchall()
            qualities = connection.execute(
                "SELECT source_id, score, peer_reviewed, primary_source, publication_date, methodology_transparency, conflict_disclosure, independence_notes FROM source_quality ORDER BY created_at DESC"
            ).fetchall()
            families = connection.execute(
                "SELECT source_id, family_id, match_type, similarity FROM source_families ORDER BY created_at DESC"
            ).fetchall()
            contradictions = connection.execute(
                "SELECT claim_a, claim_b, relation, explanation, score FROM contradiction_checks ORDER BY score DESC LIMIT 20"
            ).fetchall()
            document_analyses = connection.execute(
                "SELECT document_id, inventory, section_summary, synthesis, verification_status FROM document_analyses ORDER BY created_at DESC"
            ).fetchall()
            retrieval_rows = connection.execute(
                "SELECT term, tf, document_id FROM retrieval_index ORDER BY tf DESC LIMIT 20"
            ).fetchall()
            benchmarks = connection.execute(
                "SELECT question, unsupported_claim_rate, citation_accuracy, contradiction_detection FROM benchmark_runs ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            questions = connection.execute(
                "SELECT id, question, primary_question, subquestions, date_range, geographic_scope, population, definitions, inclusion_criteria, exclusion_criteria, source_types FROM research_questions ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            plans = connection.execute(
                "SELECT question_id, search_terms, source_types, strategy FROM source_plans ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            gaps = connection.execute(
                "SELECT question_id, gap_text, severity, recommendation FROM evidence_gaps ORDER BY created_at DESC LIMIT 20"
            ).fetchall()

        quality_lookup = {
            row["source_id"]: {
                "score": row["score"],
                "peer_reviewed": bool(row["peer_reviewed"]),
                "primary_source": bool(row["primary_source"]),
                "publication_date": row["publication_date"],
                "methodology_transparency": row["methodology_transparency"],
                "conflict_disclosure": bool(row["conflict_disclosure"]),
                "independence_notes": row["independence_notes"],
            }
            for row in qualities
        }
        family_lookup = {}
        for row in families:
            family_lookup.setdefault(row["source_id"], []).append(
                {
                    "family_id": row["family_id"],
                    "match_type": row["match_type"],
                    "similarity": row["similarity"],
                }
            )

        summary_lines = [
            f"# {title}",
            "",
            "## Executive summary",
            "",
            f"This report is based on {len(sources)} source(s) and {len(claims)} extracted claim(s). It follows an accuracy-first workflow where each substantive finding is tied back to a specific section and evidence span.",
            "",
            "## Scope and definitions",
            "",
            "The source material was ingested locally, sectioned, and stored with provenance. The system separates evidence from interpretation and requires explicit support before a conclusion is treated as a finding.",
            "",
            "## Sources consulted",
            "",
        ]
        for source in sources:
            quality = quality_lookup.get(source["id"])
            family = family_lookup.get(source["id"], [])
            quality_text = ""
            if quality:
                quality_text = (
                    f" | quality={quality['score']:.2f} | peer_reviewed={str(quality['peer_reviewed']).lower()} | "
                    f"primary={str(quality['primary_source']).lower()}"
                )
            if family:
                family_text = "; ".join(
                    f"{entry['match_type']}={entry['similarity']:.2f}" for entry in family[:2]
                )
                quality_text += f" | duplicate_family={family_text}"
            summary_lines.append(f"- {source['title']} ({source['source_type']}) — hash: {source['content_hash'][:12]}{quality_text}")

        summary_lines.extend(["", "## Main findings", ""])
        if not claims:
            summary_lines.append("No claim-level evidence was extracted from the current corpus.")
        else:
            for idx, claim in enumerate(claims, start=1):
                summary_lines.append(f"{idx}. {claim['text']}")
                summary_lines.append(f"   - Type: {claim['claim_type']} | Confidence: {claim['confidence']:.2f} | Evidence: {claim['location']}")

        summary_lines.extend(["", "## Evidence ledger", ""])
        for idx, claim in enumerate(claims, start=1):
            summary_lines.append(f"### Claim {idx}")
            summary_lines.append(f"- Text: {claim['text']}")
            summary_lines.append(f"- Type: {claim['claim_type']}")
            summary_lines.append(f"- Confidence: {claim['confidence']:.2f}")
            summary_lines.append(f"- Location: {claim['location']}")
            summary_lines.append("")

        summary_lines.extend(["", "## Contradiction analysis", ""])
        if not contradictions:
            summary_lines.append("No significant contradiction patterns were detected in the current claim set.")
        else:
            for idx, contradiction in enumerate(contradictions, start=1):
                summary_lines.append(f"{idx}. {contradiction['relation']} (score={contradiction['score']:.2f})")
                summary_lines.append(f"   - Claim A: {contradiction['claim_a']}")
                summary_lines.append(f"   - Claim B: {contradiction['claim_b']}")
                summary_lines.append(f"   - Explanation: {contradiction['explanation']}")

        summary_lines.extend(["", "## Verification pass", ""])
        verification = self.verify_claims(statements=[claim["text"] for claim in claims])
        if verification:
            supported = sum(1 for item in verification if item["supported"])
            summary_lines.append(f"Verification reviewed {len(verification)} statements and found {supported} supported by direct evidence language.")
            for idx, result in enumerate(verification[:5], start=1):
                summary_lines.append(f"{idx}. Supported={bool(result['supported'])} | {result['statement']}")

        summary_lines.extend(["", "## Retrieval index", ""])
        if not retrieval_rows:
            summary_lines.append("No retrieval index entries have been built yet.")
        else:
            for row in retrieval_rows[:10]:
                summary_lines.append(f"- term='{row['term']}' tf={row['tf']:.3f} document={row['document_id']}")

        summary_lines.extend(["", "## Document analyses", ""])
        if not document_analyses:
            summary_lines.append("No document analyses recorded for the current corpus.")
        else:
            for row in document_analyses[:5]:
                summary_lines.append(f"- document={row['document_id']} status={row['verification_status']}")
                summary_lines.append(f"  summary: {row['synthesis']}")

        summary_lines.extend(["", "## Benchmarking", ""])
        if not benchmarks:
            summary_lines.append("No benchmark runs recorded yet.")
        else:
            for row in benchmarks[:5]:
                summary_lines.append(
                    f"- question='{row['question']}' unsupported={row['unsupported_claim_rate']:.2f} citation={row['citation_accuracy']:.2f} contradiction={row['contradiction_detection']:.2f}"
                )

        summary_lines.extend(["", "## Research questions and plans", ""])
        if not questions:
            summary_lines.append("No research questions or source plans have been defined yet.")
        else:
            for row in questions[:5]:
                summary_lines.append(f"- question='{row['question']}' primary='{row['primary_question']}'")
                if row["subquestions"]:
                    summary_lines.append(f"  subquestions: {row['subquestions']}")
            if plans:
                for row in plans[:5]:
                    summary_lines.append(f"  plan: {row['strategy']} | terms={row['search_terms']} | sources={row['source_types']}")

        summary_lines.extend(["", "## Evidence gaps", ""])
        if not gaps:
            summary_lines.append("No evidence gaps have been recorded yet.")
        else:
            for row in gaps[:5]:
                summary_lines.append(f"- severity={row['severity']} | question={row['question_id']} | {row['gap_text']}")
                if row["recommendation"]:
                    summary_lines.append(f"  recommendation: {row['recommendation']}")

        markdown = "\n".join(summary_lines).strip() + "\n"
        report_id = f"report-{uuid.uuid4().hex[:8]}"
        report_path = self.exports_dir / f"{report_id}.md"
        report_path.write_text(markdown, encoding="utf-8")

        with self._connect() as connection:
            connection.execute(
                "INSERT INTO reports (id, title, markdown, created_at) VALUES (?, ?, ?, ?)",
                (report_id, title, markdown, _utc_now()),
            )
            connection.commit()

        return markdown


def create_project(root: str | Path) -> ResearchProject:
    project = ResearchProject(root)
    project.create()
    return project
