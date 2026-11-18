from copy import deepcopy
import json
from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG, QueryParam
from lightrag.evidence import EvidenceCheckError, EvidenceCitation, check_query_evidence


pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

MATERIALS = "What documents are required for lodging reimbursement?"
CAP = "What is the nightly lodging cap?"
QUERY = "What documents are required and what is the nightly lodging cap?"
QUOTE = "a hotel invoice and a check-in list"
RETRIEVAL = {
    "status": "success",
    "data": {"chunks": [{"chunk_id": "lodging", "content": f"Submit {QUOTE}."}]},
}


@pytest.fixture
def query_stubs(monkeypatch):
    retrieval = deepcopy(RETRIEVAL)
    retrieval["data"]["chunks"][0]["file_path"] = "handbook.md"
    retrieval["data"]["chunks"].append(
        {
            "chunk_id": "distractor",
            "content": "Flights cost 500.",
            "file_path": "travel.md",
        }
    )
    retrieve = AsyncMock(return_value=retrieval)
    generate = AsyncMock(return_value="Bring a hotel invoice and a check-in list.")
    monkeypatch.setattr(LightRAG, "aquery_data", retrieve)
    monkeypatch.setattr(
        LightRAG,
        "_build_global_config",
        lambda self: {"role_llm_funcs": {"query": generate}},
    )
    return object.__new__(LightRAG), retrieve, generate


@pytest.mark.parametrize("partial", [False, True])
async def test_evidence_query_generates_from_accepted_quotes_only(query_stubs, partial):
    rag, retrieve, generate = query_stubs
    query = QUERY if partial else MATERIALS
    citations = {MATERIALS: [{"chunk_id": "lodging", "quote": QUOTE}]}
    if partial:
        citations[CAP] = []
    checker = AsyncMock(return_value=citations)
    progress = AsyncMock()
    result = await rag.aquery_llm(
        query,
        evidence_checker=checker,
        evidence_subquestions=list(citations),
        progress_callback=progress,
    )
    retrieve.assert_awaited_once()
    checker.assert_awaited_once()
    generate.assert_awaited_once()
    progress.assert_awaited_once()
    assert json.loads(generate.await_args.args[0]) == [
        {"question": MATERIALS, "quotes": [QUOTE]}
    ]
    assert result["evidence"]["status"] == ("partial" if partial else "sufficient")
    assert (
        result["evidence"]["questions"][0]["citations"][0]["file_path"] == "handbook.md"
    )
    text = result["llm_response"]["content"]
    assert "handbook.md" in text
    assert (CAP in text) is partial
    assert "travel.md" not in text


@pytest.mark.parametrize("no_results", [False, True])
async def test_evidence_query_skips_generation_without_support(query_stubs, no_results):
    rag, retrieve, generate = query_stubs
    if no_results:
        retrieve.return_value = {
            "status": "failure",
            "data": {},
            "metadata": {"failure_reason": "no_results"},
        }
    checker = AsyncMock(return_value={CAP: []})
    result = await rag.aquery_llm(CAP, evidence_checker=checker)
    retrieve.assert_awaited_once()
    generate.assert_not_called()
    assert checker.await_count == (0 if no_results else 1)
    assert result["evidence"]["status"] == "insufficient"
    assert CAP in result["llm_response"]["content"]
    assert not result["llm_response"]["llm_generated"]


async def test_evidence_query_reports_checker_failure(query_stubs):
    rag, retrieve, generate = query_stubs
    result = await rag.aquery_llm(
        CAP, evidence_checker=AsyncMock(side_effect=TimeoutError)
    )
    retrieve.assert_awaited_once()
    generate.assert_not_called()
    assert result["status"] == "failure"
    assert result["metadata"]["failure_reason"] == "evidence_check_failed"


async def test_evidence_query_default_uses_original_query_path(query_stubs):
    rag, retrieve, generate = query_stubs
    result = await rag.aquery_llm(CAP, QueryParam(mode="bypass", stream=False))
    retrieve.assert_not_called()
    generate.assert_awaited_once()
    assert generate.await_args.args == (CAP,)
    assert "evidence" not in result


@pytest.mark.parametrize(
    "param",
    [
        QueryParam(mode="bypass"),
        QueryParam(stream=True),
        QueryParam(only_need_context=True),
        QueryParam(only_need_prompt=True),
    ],
)
async def test_evidence_query_rejects_incompatible_modes(query_stubs, param):
    rag, retrieve, generate = query_stubs
    with pytest.raises(ValueError, match="non-streaming RAG"):
        await rag.aquery_llm(CAP, param, evidence_checker=AsyncMock())
    retrieve.assert_not_called()
    generate.assert_not_called()


async def test_sufficient_evidence_returns_validated_quote():
    async def checker(question, questions, chunks):
        assert question == MATERIALS
        assert questions == (MATERIALS,)
        assert chunks[0].content == f"Submit {QUOTE}."
        return {MATERIALS: [{"chunk_id": chunks[0].chunk_id, "quote": QUOTE}]}

    result = await check_query_evidence(MATERIALS, RETRIEVAL, checker)

    assert result.status == "sufficient"
    assert result.questions[0].citations == (EvidenceCitation("lodging", QUOTE),)


async def test_partial_evidence_preserves_supported_and_missing_questions():
    checker = AsyncMock(
        return_value={CAP: [], MATERIALS: [{"chunk_id": "lodging", "quote": QUOTE}]}
    )
    result = await check_query_evidence(
        QUERY, RETRIEVAL, checker, subquestions=[MATERIALS, CAP]
    )

    assert result.status == "partial"
    assert [item.question for item in result.questions] == [MATERIALS, CAP]
    assert result.questions[0].citations == (EvidenceCitation("lodging", QUOTE),)
    assert result.questions[1].citations == ()


async def test_related_chunk_can_be_insufficient():
    checker = AsyncMock(return_value={CAP: []})

    result = await check_query_evidence(CAP, RETRIEVAL, checker)

    assert result.status == "insufficient"
    assert result.questions[0].citations == ()
    checker.assert_awaited_once()


@pytest.mark.parametrize(
    "retrieval",
    [
        {"status": "success", "data": {"chunks": []}},
        {"status": "failure", "metadata": {"failure_reason": "no_results"}, "data": {}},
    ],
)
async def test_empty_retrieval_skips_checker(retrieval):
    checker = AsyncMock()

    result = await check_query_evidence(CAP, retrieval, checker)

    assert result.status == "insufficient"
    assert result.questions[0].question == CAP
    checker.assert_not_awaited()


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {CAP: [{"chunk_id": "not-retrieved", "quote": QUOTE}]},
            "outside this retrieval",
        ),
        ({CAP: [{"chunk_id": "lodging", "quote": "500 yuan"}]}, "does not match"),
        ({CAP: [{"chunk_id": "lodging", "quote": " "}]}, "does not match"),
        ({CAP: "missing"}, "sequence of citations"),
        ({}, "every requested question"),
        ({CAP: [], MATERIALS: []}, "every requested question"),
    ],
)
async def test_invalid_checker_output_is_a_failure(payload, error):
    checker = AsyncMock(return_value=payload)

    with pytest.raises(EvidenceCheckError, match=error):
        await check_query_evidence(CAP, RETRIEVAL, checker)


async def test_checker_failure_is_distinct_from_insufficient_evidence():
    cause = TimeoutError("checker timed out")
    checker = AsyncMock(side_effect=cause)

    with pytest.raises(EvidenceCheckError, match="checker failed") as error:
        await check_query_evidence(CAP, RETRIEVAL, checker)

    assert error.value.__cause__ is cause


@pytest.mark.parametrize(
    "retrieval",
    [
        {"status": "failure", "message": "storage unavailable"},
        {"status": "success", "data": {}},
        {"status": "success", "data": {"chunks": [{"content": "Missing ID"}]}},
        {
            "status": "success",
            "data": {
                "chunks": [
                    {"chunk_id": "same", "content": "First text"},
                    {"chunk_id": "same", "content": "Conflicting text"},
                ]
            },
        },
    ],
)
async def test_invalid_retrieval_does_not_reach_checker(retrieval):
    checker = AsyncMock()

    with pytest.raises(EvidenceCheckError):
        await check_query_evidence(CAP, retrieval, checker)

    checker.assert_not_awaited()
