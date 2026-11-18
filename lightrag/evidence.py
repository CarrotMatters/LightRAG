"""Validate evidence reported by an optional checker over aquery_data results."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from typing import Literal


class EvidenceCheckError(RuntimeError):
    """The retrieval response or checker could not yield a valid assessment."""


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    content: str


@dataclass(frozen=True)
class EvidenceCitation:
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class QuestionEvidence:
    question: str
    citations: tuple[EvidenceCitation, ...]


@dataclass(frozen=True)
class QueryEvidence:
    status: Literal["sufficient", "partial", "insufficient"]
    questions: tuple[QuestionEvidence, ...]


EvidenceChecker = Callable[
    [str, tuple[str, ...], tuple[EvidenceChunk, ...]],
    Awaitable[Mapping[str, Sequence[Mapping[str, str]]]],
]


def _read_chunks(retrieval: Mapping) -> tuple[EvidenceChunk, ...]:
    if not isinstance(retrieval, Mapping):
        raise EvidenceCheckError("Expected a structured retrieval response")

    status = retrieval.get("status")
    if status == "failure":
        metadata = retrieval.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("failure_reason") == "no_results"
        ):
            return ()
        raise EvidenceCheckError("Retrieval failed before evidence could be checked")

    data = retrieval.get("data")
    if status != "success" or not isinstance(data, Mapping):
        raise EvidenceCheckError("Invalid structured retrieval response")
    raw_chunks = data.get("chunks")
    if not isinstance(raw_chunks, list):
        raise EvidenceCheckError(
            "Structured retrieval response must contain a chunk list"
        )

    chunks = {}
    for raw in raw_chunks:
        if not isinstance(raw, Mapping):
            raise EvidenceCheckError("Invalid retrieved chunk")
        chunk_id, content = raw.get("chunk_id"), raw.get("content")
        if (
            not isinstance(chunk_id, str)
            or not chunk_id.strip()
            or not isinstance(content, str)
        ):
            raise EvidenceCheckError(
                "Retrieved chunks require a chunk ID and text content"
            )
        if chunk_id in chunks and chunks[chunk_id].content != content:
            raise EvidenceCheckError(
                "Retrieved chunks disagree on content for one chunk ID"
            )
        chunks[chunk_id] = EvidenceChunk(chunk_id, content)
    return tuple(chunks.values())


def _validate_citations(
    citations: Sequence[Mapping[str, str]], chunks_by_id: Mapping[str, str]
) -> tuple[EvidenceCitation, ...]:
    if not isinstance(citations, Sequence) or isinstance(citations, (str, bytes)):
        raise EvidenceCheckError("Each question requires a sequence of citations")

    validated = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise EvidenceCheckError("Each citation requires a chunk ID and quote")
        chunk_id, quote = citation.get("chunk_id"), citation.get("quote")
        if not isinstance(chunk_id, str) or chunk_id not in chunks_by_id:
            raise EvidenceCheckError(
                "Citation refers to a chunk outside this retrieval"
            )
        if (
            not isinstance(quote, str)
            or not quote.strip()
            or quote not in chunks_by_id[chunk_id]
        ):
            raise EvidenceCheckError(
                "Citation quote does not match the retrieved chunk"
            )
        validated.append(EvidenceCitation(chunk_id, quote))
    return tuple(validated)


async def check_query_evidence(
    question: str,
    retrieval: Mapping,
    checker: EvidenceChecker,
    *,
    subquestions: Sequence[str] | None = None,
) -> QueryEvidence:
    """Assess retrieved text for the question or explicitly supplied subquestions.

    Pass the full result from ``await rag.aquery_data(...)``. The checker receives
    the original question, the ordered questions to assess, and the retrieved
    chunks. It returns one citation sequence per question, keyed by the exact
    question text. An empty sequence reports missing evidence for that question.

    The checker judges semantic support. This function verifies that its citations
    use retrieved IDs and verbatim quotes, then derives the overall status.
    A retrieval with no chunks skips the checker. Other retrieval failures, checker
    exceptions, and malformed responses raise ``EvidenceCheckError``.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string")
    if isinstance(subquestions, (str, bytes)):
        raise ValueError("Subquestions must be a sequence of question strings")
    questions = (question,) if subquestions is None else tuple(subquestions)
    if not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Subquestions must contain non-empty strings")
    if len(set(questions)) != len(questions):
        raise ValueError("Subquestions must be unique")

    chunks = _read_chunks(retrieval)
    if not chunks:
        return QueryEvidence(
            "insufficient", tuple(QuestionEvidence(q, ()) for q in questions)
        )

    try:
        response = await checker(question, questions, chunks)
    except Exception as exc:
        raise EvidenceCheckError("Evidence checker failed") from exc
    if not isinstance(response, Mapping) or response.keys() != set(questions):
        raise EvidenceCheckError(
            "Checker must assess every requested question exactly once"
        )

    chunks_by_id = {chunk.chunk_id: chunk.content for chunk in chunks}
    assessments = tuple(
        QuestionEvidence(q, _validate_citations(response[q], chunks_by_id))
        for q in questions
    )
    if all(item.citations for item in assessments):
        return QueryEvidence("sufficient", assessments)
    if any(item.citations for item in assessments):
        return QueryEvidence("partial", assessments)
    return QueryEvidence("insufficient", assessments)


async def answer_with_evidence(
    assessment: QueryEvidence,
    retrieval: Mapping,
    generate: Callable[[str], Awaitable[str]],
) -> dict:
    """Build an answer using only validated quotes and attach their provenance."""
    sources = {
        chunk["chunk_id"]: chunk.get("file_path")
        for chunk in retrieval.get("data", {}).get("chunks", [])
    }
    questions = [
        {
            "question": item.question,
            "citations": [
                {
                    "chunk_id": cite.chunk_id,
                    "quote": cite.quote,
                    "file_path": sources[cite.chunk_id],
                }
                for cite in item.citations
            ],
        }
        for item in assessment.questions
    ]
    supported = [item for item in questions if item["citations"]]
    missing = [item["question"] for item in questions if not item["citations"]]
    parts = []
    if supported:
        prompt = json.dumps(
            [
                {
                    "question": item["question"],
                    "quotes": [cite["quote"] for cite in item["citations"]],
                }
                for item in supported
            ],
            ensure_ascii=False,
        )
        answer = await generate(prompt)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("The query model did not return a non-empty answer")
        parts.append(answer.strip())
        references = []
        for item in supported:
            for cite in item["citations"]:
                source = cite["file_path"] or cite["chunk_id"]
                references.append(f"- {item['question']} — {source}: {cite['quote']}")
        parts.append("Sources\n" + "\n".join(references))
    if missing:
        parts.append(
            "Insufficient evidence for:\n" + "\n".join(f"- {q}" for q in missing)
        )
    return {
        **retrieval,
        "evidence": {"status": assessment.status, "questions": questions},
        "llm_response": {
            "content": "\n\n".join(parts),
            "response_iterator": None,
            "is_streaming": False,
            "llm_generated": bool(supported),
        },
    }
