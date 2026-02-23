# Risk Map - Hermes

This document maps each source module to its test coverage quality, identifies the highest-risk code paths, and provides a prioritized remediation plan.

---

## Module Risk Assessment

| Module | Risk Level | Test Coverage | Notes |
|--------|-----------|---------------|-------|
| `embedding_provider.py` | **HIGH** | Partial | SentenceTransformerProvider has 3 known bugs (C1, C2, C3). OpenAI batch embed sends wrong key. Tests are skipped or missing for these paths. |
| `llm.py` | **MEDIUM-HIGH** | Partial | Provider detection/config well-tested. OpenAI HTTP call normalization, error hierarchy, and _estimate_usage untested. |
| `main.py` | **MEDIUM** | Fair | API endpoints covered for happy paths and basic validation. Sophia context retrieval has gap in timeout/non-201 paths. Media ingestion test uses wrong field name. |
| `services.py` | **MEDIUM** | Indirect only | All testing goes through API layer. No direct unit tests for generate_embedding, generate_embeddings_batch, process_nlp. Milvus persistence is fire-and-forget, failures silently swallowed. |
| `proposal_builder.py` | **LOW** | Good | Well-tested with mocks. Pipeline orchestration, graceful degradation, and metadata population all covered. |
| `context_cache.py` | **LOW** | Good | All paths covered: happy path, Redis errors, unavailability. Queue structure verified. |
| `ner_provider.py` | **LOW** | Good | Both spaCy and OpenAI providers tested. _parse_response has thorough edge case coverage. Factory and singleton tested. |
| `relation_extractor.py` | **MEDIUM** | Partial | SpacyRelationExtractor well-tested. OpenAIRelationExtractor._parse_response has **zero tests** despite complex validation logic. |
| `env.py` | **LOW-MEDIUM** | Minimal | No direct tests for load_env_file, get_milvus_config, get_neo4j_config. These are exercised indirectly but quote-stripping and fallback logic untested. |
| `milvus_client.py` | **LOW** | Good | connect, disconnect, ensure_collection, persist_embedding, initialize all tested with mocks. |

---

## Highest-Risk Code Paths

### 1. OpenAI Embedding Batch Pipeline (CRITICAL)

```
ProposalBuilder.build()
  -> _run_batch_embed()
    -> generate_embeddings_batch()  [services.py]
      -> provider.embed_batch()     [embedding_provider.py]
        -> payload = {"XXmodelXX": ...}  <-- BUG: wrong key
```

Every proposal built through the cognitive loop will fail when using OpenAI embeddings because `embed_batch` sends `"XXmodelXX"` instead of `"model"`.

**Impact:** Complete proposal pipeline failure for OpenAI embedding users.

### 2. SentenceTransformerProvider Initialization (CRITICAL)

```
get_embedding_provider()
  -> SentenceTransformerProvider(model="all-MiniLM-L6-v2")
    -> self._model_name = None  <-- BUG: ignores model arg
    -> _load() -> SentenceTransformer(None)  <-- will crash or load wrong model
```

The sentence-transformers backend will either crash or load a default model that may not match the configured one.

**Impact:** Silent model mismatch or crash on first embedding request.

### 3. Sophia Context Retrieval Resilience

```
_get_sophia_context()
  -> Redis cache check (tested)
  -> Sophia HTTP call
    -> ConnectError (tested)
    -> TimeoutException (UNTESTED)
    -> Non-201 response (UNTESTED)
    -> Generic exception (UNTESTED)
```

While all paths return `[]` on failure (never raises), the specific logging and behavior under timeout vs. error has no verification.

**Impact:** Low immediate risk (graceful degradation), but logging and observability gaps make production debugging harder.

### 4. LLM Response Normalization

```
OpenAIProvider.generate()
  -> response.json()
  -> _normalize_choices(data.get("choices"))
    -> "text" fallback path (UNTESTED)
    -> empty choices fallback (UNTESTED)
```

If an OpenAI-compatible provider returns unexpected response shapes, the normalization logic may silently produce wrong data.

**Impact:** Incorrect LLM responses passed to downstream consumers.

---

## Test Type Distribution

| Test Type | File Count | Approximate Line Count | Value Assessment |
|-----------|-----------|----------------------|-----------------|
| Unit tests (mocked) | 8 files | ~1,100 lines | **HIGH** -- these catch real bugs |
| API integration (TestClient) | 4 files | ~900 lines | **MEDIUM** -- validate contract but often too permissive |
| Infrastructure integration | 3 files | ~1,200 lines | **LOW** -- rarely run, mostly test external systems |
| Performance tests | 1 file | ~500 lines | **LOW** -- measure timing, don't catch bugs |
| UI tests | 1 file | ~60 lines | **LOW** -- minimal coverage of a minor feature |

**Recommendation:** Shift ~400 lines of infrastructure integration testing effort into unit tests for `services.py`, `llm.py` (OpenAI HTTP path), and `relation_extractor.py` (parse_response).

---

## Prioritized Remediation Plan

### Priority 1: Fix and test the three bugs in embedding_provider.py
1. Fix `SentenceTransformerProvider.__init__` to assign `self._model_name = model`
2. Fix `SentenceTransformerProvider.dimension` to call `model.get_sentence_embedding_dimension()`
3. Fix `embed_batch` to use `"model"` key instead of `"XXmodelXX"`
4. Un-skip `TestSentenceTransformerProvider` and add `test_embed_batch_payload`
5. Add a test that verifies `embed_batch` sends the correct `"model"` key in the payload

### Priority 2: Add unit tests for untested critical paths
6. Test `OpenAIRelationExtractor._parse_response` with: valid JSON, code-fenced JSON, invalid JSON, entity name validation, confidence clamping, deduplication, relation normalization
7. Test `_normalize_choices` with the `"text"` fallback and empty choices
8. Test `_estimate_usage` with empty strings
9. Fix the media ingestion test to use the correct field name `"file"` instead of `"media"`

### Priority 3: Improve assertion specificity
10. Replace `assert status_code in [200, 400, 500, 503]` patterns with specific expected codes
11. Add a test for `LLMProviderNotConfiguredError` that specifically asserts 503
12. Add timeout and non-201 response tests for `_get_sophia_context`

### Priority 4: Run mutation testing
13. Run mutmut against `embedding_provider.py`, `llm.py`, `relation_extractor.py`, `proposal_builder.py`
14. Analyze surviving mutants and write targeted tests to kill them
15. Target 80% mutation kill rate

---

## Module Dependency Graph (risk flows downstream)

```
main.py (API layer)
  |
  +-> services.py (orchestration)
  |     |
  |     +-> embedding_provider.py  [HIGH RISK - 3 bugs]
  |     +-> milvus_client.py       [OK]
  |     +-> llm.py                 [MEDIUM RISK]
  |
  +-> proposal_builder.py (cognitive loop)
  |     |
  |     +-> ner_provider.py        [OK]
  |     +-> relation_extractor.py  [MEDIUM RISK - parse untested]
  |     +-> embedding_provider.py  [HIGH RISK]
  |     +-> services.py
  |
  +-> context_cache.py             [OK]
  |
  +-> env.py                       [LOW-MEDIUM RISK]
```

Risk propagates upward: bugs in `embedding_provider.py` affect both the `/embed_text` endpoint and the entire proposal pipeline.
