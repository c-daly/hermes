"""Builds structured graph-ready proposals from conversational text.

Hermes is the only component that understands language. The ProposalBuilder
translates text into structured data (entities + embeddings) that Sophia
can process without reading text.
"""

import asyncio
import logging
import uuid as uuid_mod
from datetime import UTC, datetime

from hermes.relation_extractor import get_relation_extractor
from hermes.services import generate_embedding, process_nlp

logger = logging.getLogger(__name__)

# Map spaCy NER labels to ontology type definitions.
SPACY_TO_ONTOLOGY: dict[str, str] = {
    "GPE": "location",
    "LOC": "location",
    "FAC": "object",
    "ORG": "agent",
    "PERSON": "agent",
    "NORP": "entity",
    "PRODUCT": "object",
    "EVENT": "process",
    "WORK_OF_ART": "entity",
    "LAW": "concept",
    "LANGUAGE": "concept",
    "DATE": "state",
    "TIME": "state",
    "QUANTITY": "data",
    "CARDINAL": "data",
    "ORDINAL": "data",
    "MONEY": "data",
    "PERCENT": "data",
}


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
        """
        proposal_id = str(uuid_mod.uuid4())
        now = datetime.now(UTC).isoformat()

        proposed_nodes = await self._extract_entities(text)
        proposed_edges = await self._extract_relations(text, proposed_nodes)
        document_embedding = await self._generate_document_embedding(text)

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
            "metadata": metadata,
        }

    async def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities and generate per-entity embeddings."""
        try:
            nlp_result = await process_nlp(text, ["ner"])
            entities = nlp_result.get("entities", [])
        except Exception:
            logger.warning("NER extraction failed, returning empty entities")
            return []

        async def _process_entity(entity: dict) -> dict:
            emb = await generate_embedding(entity["text"])
            ontology_type = SPACY_TO_ONTOLOGY.get(entity["label"], "entity")
            return {
                "name": entity["text"],
                "type": ontology_type,
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
                    entity.get("text", "<unknown>"),
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
            phrase = f"{edge['source_name']} {edge['relation'].lower().replace('_', ' ')} {edge['target_name']}"
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
