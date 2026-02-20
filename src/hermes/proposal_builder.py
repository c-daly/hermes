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
from hermes.services import generate_embedding

logger = logging.getLogger(__name__)


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
        """
        proposal_id = str(uuid_mod.uuid4())
        now = datetime.now(UTC).isoformat()

        t0 = time.monotonic()

        proposed_nodes = await self._extract_entities(text)
        t_ner = time.monotonic()

        proposed_edges = await self._extract_relations(text, proposed_nodes)
        t_rel = time.monotonic()

        document_embedding = await self._generate_document_embedding(text)
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
            "document_embedding": document_embedding,
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
            emb = get_embedding_provider()
            embedding_provider_name = emb.model_name
        except Exception:
            embedding_provider_name = "unknown"

        return {
            "ner_provider": ner_provider_name,
            "embedding_provider": embedding_provider_name,
            "ner_duration_ms": ner_duration_ms,
            "relation_duration_ms": relation_duration_ms,
            "embedding_duration_ms": embedding_duration_ms,
            "total_duration_ms": total_duration_ms,
            "entity_count": entity_count,
            "edge_count": edge_count,
        }

    async def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities and generate per-entity embeddings."""
        try:
            provider = get_ner_provider()
            entities = await provider.extract_entities(text)
        except Exception:
            logger.warning("NER extraction failed, returning empty entities")
            return []

        async def _process_entity(entity: dict) -> dict:
            emb = await generate_embedding(entity["name"])
            return {
                "name": entity["name"],
                "type": entity["type"],
                "embedding": emb["embedding"],
                "embedding_id": emb["embedding_id"],
                "dimension": emb["dimension"],
                "model": emb["model"],
                "properties": {"start": entity["start"], "end": entity["end"]},
            }

        results = await asyncio.gather(
            *[_process_entity(e) for e in entities],
            return_exceptions=True,
        )
        processed = []
        for entity, result in zip(entities, results):
            if isinstance(result, dict):
                processed.append(result)
            elif isinstance(result, Exception):
                logger.warning(
                    "Failed to process entity '%s': %s",
                    entity.get("name", "<unknown>"),
                    result,
                )
        return processed

    async def _extract_relations(
        self, text: str, proposed_nodes: list[dict]
    ) -> list[dict]:
        """Extract relations between entities and embed relation phrases."""
        if len(proposed_nodes) < 2:
            return []

        try:
            extractor = get_relation_extractor()
            raw_edges = await extractor.extract(text, proposed_nodes)
        except Exception:
            logger.warning("Relation extraction failed, returning no edges")
            return []

        async def _embed_edge(edge: dict) -> dict:
            phrase = f"{edge.get('source_name', '')} {edge.get('relation', 'RELATED_TO').lower().replace('_', ' ')} {edge.get('target_name', '')}"
            emb = await generate_embedding(phrase)
            edge["embedding"] = emb["embedding"]
            edge["model"] = emb["model"]
            return edge

        results = await asyncio.gather(
            *[_embed_edge(e) for e in raw_edges],
            return_exceptions=True,
        )
        processed: list[dict] = []
        for edge, result in zip(raw_edges, results):
            if isinstance(result, dict):
                processed.append(result)
            elif isinstance(result, Exception):
                logger.warning(
                    "Failed to embed edge %s->%s: %s",
                    edge.get("source_name"),
                    edge.get("target_name"),
                    result,
                )
        return processed

    async def _generate_document_embedding(self, text: str) -> dict | None:
        """Generate embedding for the full text."""
        try:
            return await generate_embedding(text)
        except Exception:
            logger.warning("Document embedding generation failed")
            return None
