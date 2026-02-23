"""Builds structured graph-ready proposals from conversational text.

Hermes is the only component that understands language. The ProposalBuilder
translates text into structured data (entities + embeddings) that Sophia
can process without reading text.
"""

import asyncio
import logging
import time
import uuid as uuid_mod
from datetime import UTC, datetime

from hermes.embedding_provider import get_embedding_provider
from hermes.ner_provider import get_ner_provider
from hermes.relation_extractor import get_relation_extractor
from hermes.services import generate_embeddings_batch

logger = logging.getLogger(__name__)

try:
    from logos_observability import get_tracer

    tracer = get_tracer("hermes.proposal_builder")
except ImportError:
    from contextlib import nullcontext

    from typing import Any

    class _NoopTracer:
        def start_as_current_span(self, name: str, **kw: Any) -> nullcontext:  # type: ignore[type-arg]
            return nullcontext()

    tracer: Any = _NoopTracer()  # type: ignore[no-redef]


class ProposalBuilder:
    """Builds structured proposals from conversation turns."""

    async def build(
        self,
        text: str,
        metadata: dict,
        *,
        correlation_id: str | None = None,
        llm_provider: str = "unknown",
        model: str = "unknown",
        confidence: float = 0.7,
    ) -> dict:
        """Build a structured proposal from text.

        Returns dict matching HermesProposalRequest schema.
        Injects ``metadata["pipeline"]`` with provider info and timing.

        Pipeline is parallelized where possible:
        1. NER extraction (must be first — entities needed downstream)
        2. In parallel: relation extraction + batch embed (entities + doc text)
        3. Batch embed edge phrases (needs relation extraction results)
        """
        proposal_id = str(uuid_mod.uuid4())
        now = datetime.now(UTC).isoformat()

        with tracer.start_as_current_span(
            "proposal_builder.build",
            attributes={"proposal_id": proposal_id, "text_length": len(text)},
        ):
            t0 = time.monotonic()

            # Step 1: NER — must complete before relation extraction
            with tracer.start_as_current_span("proposal_builder.ner"):
                entities = await self._run_ner(text)
            t_ner = time.monotonic()

            # Step 2: Run relation extraction and entity+doc embeddings in parallel
            # Relation extraction only needs entity names, not their embeddings.
            with tracer.start_as_current_span(
                "proposal_builder.parallel_extract_embed"
            ):
                entity_names = [e["name"] for e in entities]
                embed_texts = entity_names + [text]  # entities + document text

                rel_task = self._run_relation_extraction(text, entities)
                emb_task = self._run_batch_embed(embed_texts)
                raw_edges, all_embeddings = await asyncio.gather(rel_task, emb_task)
            t_rel = time.monotonic()

            # Unpack embeddings: first N are entities, last one is document
            entity_embeddings = all_embeddings[: len(entities)]
            if len(entity_embeddings) != len(entities):
                logger.warning(
                    "Entity embedding count mismatch: %d entities vs %d embeddings",
                    len(entities),
                    len(entity_embeddings),
                )
            doc_embedding_result = (
                all_embeddings[len(entities)]
                if len(all_embeddings) > len(entities)
                else None
            )

            # Assemble proposed_nodes with their embeddings
            proposed_nodes = []
            for entity, emb in zip(entities, entity_embeddings):
                proposed_nodes.append(
                    {
                        "name": entity["name"],
                        "type": entity["type"],
                        "embedding": emb["embedding"],
                        "embedding_id": emb["embedding_id"],
                        "dimension": emb["dimension"],
                        "model": emb["model"],
                        "properties": {"start": entity["start"], "end": entity["end"]},
                    }
                )

            # Step 3: Batch embed edge phrases (needs relation extraction results)
            with tracer.start_as_current_span("proposal_builder.edge_embedding"):
                proposed_edges = await self._embed_edges(raw_edges)
            t_emb = time.monotonic()

            # Build pipeline metadata for experiment tracking
            pipeline = self._build_pipeline_metadata(
                ner_duration_ms=round((t_ner - t0) * 1000, 1),
                relation_duration_ms=round((t_rel - t_ner) * 1000, 1),
                embedding_duration_ms=round((t_emb - t_rel) * 1000, 1),
                total_duration_ms=round((t_emb - t0) * 1000, 1),
                entity_count=len(proposed_nodes),
                edge_count=len(proposed_edges),
            )

            # Merge pipeline into metadata (preserving existing keys)
            enriched_metadata = dict(metadata) if metadata else {}
            enriched_metadata["pipeline"] = pipeline

            return {
                "proposal_id": proposal_id,
                "correlation_id": correlation_id,
                "source_service": "hermes",
                "llm_provider": llm_provider,
                "model": model,
                "generated_at": now,
                "confidence": confidence,
                "raw_text": text,
                "proposed_nodes": proposed_nodes,
                "proposed_edges": proposed_edges,
                "document_embedding": doc_embedding_result,
                "metadata": enriched_metadata,
            }

    def _build_pipeline_metadata(
        self,
        *,
        ner_duration_ms: float,
        relation_duration_ms: float,
        embedding_duration_ms: float,
        total_duration_ms: float,
        entity_count: int,
        edge_count: int,
    ) -> dict:
        """Build pipeline metadata dict with provider info and timing."""
        # Safely read provider names
        try:
            ner = get_ner_provider()
            ner_provider_name = getattr(ner, "name", type(ner).__name__)
        except Exception:
            ner_provider_name = "unknown"

        try:
            rel = get_relation_extractor()
            relation_provider_name = getattr(rel, "name", type(rel).__name__)
        except Exception:
            relation_provider_name = "unknown"

        try:
            emb = get_embedding_provider()
            embedding_provider_name = emb.model_name
        except Exception:
            embedding_provider_name = "unknown"

        return {
            "ner_provider": ner_provider_name,
            "relation_provider": relation_provider_name,
            "embedding_provider": embedding_provider_name,
            "ner_duration_ms": ner_duration_ms,
            "relation_duration_ms": relation_duration_ms,
            "embedding_duration_ms": embedding_duration_ms,
            "total_duration_ms": total_duration_ms,
            "entity_count": entity_count,
            "edge_count": edge_count,
        }

    async def _run_ner(self, text: str) -> list[dict]:
        """Run NER extraction only (no embeddings)."""
        try:
            provider = get_ner_provider()
            entities = await provider.extract_entities(text)
            return entities or []
        except Exception:
            logger.warning("NER extraction failed, returning empty entities")
            return []

    async def _run_relation_extraction(
        self, text: str, entities: list[dict]
    ) -> list[dict]:
        """Run relation extraction only (no embeddings)."""
        if len(entities) < 2:
            return []
        try:
            extractor = get_relation_extractor()
            return await extractor.extract(text, entities)
        except Exception:
            logger.warning("Relation extraction failed, returning no edges")
            return []

    async def _run_batch_embed(self, texts: list[str]) -> list[dict]:
        """Batch embed a list of texts in a single API call."""
        if not texts:
            return []
        try:
            return await generate_embeddings_batch(texts)
        except Exception:
            logger.warning("Batch embedding failed")
            return []

    async def _embed_edges(self, raw_edges: list[dict]) -> list[dict]:
        """Batch embed edge phrases and attach to edge dicts."""
        if not raw_edges:
            return []

        phrases = [
            f"{e['source_name']} {e['relation'].lower().replace('_', ' ')} {e['target_name']}"
            for e in raw_edges
        ]
        try:
            embeddings = await generate_embeddings_batch(phrases)
        except Exception:
            logger.warning(
                "Batch edge embedding failed, returning edges without embeddings"
            )
            return raw_edges

        if len(embeddings) != len(raw_edges):
            logger.warning(
                "Edge embedding count mismatch: %d edges vs %d embeddings",
                len(raw_edges),
                len(embeddings),
            )
            return raw_edges

        for edge, emb in zip(raw_edges, embeddings):
            edge["embedding"] = emb["embedding"]
            edge["model"] = emb["model"]

        return raw_edges
