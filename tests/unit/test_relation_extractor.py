"""Tests for relation_extractor module."""

import pytest
from unittest.mock import MagicMock, patch

pytestmark = pytest.mark.unit


class TestSpacyRelationExtractor:
    """Tests for the spaCy-based relation extractor."""

    @pytest.mark.asyncio
    async def test_returns_empty_for_fewer_than_two_entities(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()
        result = await extractor.extract("Hello", [{"name": "Hello"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_entities(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()
        result = await extractor.extract("Hello world", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_extracts_verb_relation(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()

        # Build mock spaCy doc with two entities and a verb between them
        mock_ent1 = MagicMock()
        mock_ent1.text = "Alice"
        mock_ent1.start = 0
        mock_ent1.end = 1
        mock_ent1.root = MagicMock()
        mock_ent1.root.head = MagicMock(pos_="NOUN")

        mock_ent2 = MagicMock()
        mock_ent2.text = "Google"
        mock_ent2.start = 3
        mock_ent2.end = 4
        mock_ent2.root = MagicMock()
        mock_ent2.root.head = MagicMock(pos_="NOUN")

        # Verb token between entities
        verb_tok = MagicMock()
        verb_tok.pos_ = "VERB"
        verb_tok.lemma_ = "work"
        verb_tok.text = "works"

        # Build mock sent
        mock_sent = MagicMock()
        mock_sent.ents = [mock_ent1, mock_ent2]

        # Build mock doc
        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_doc.__getitem__ = lambda self, s: (
            [verb_tok] if isinstance(s, slice) else verb_tok
        )
        mock_doc.__len__ = lambda self: 5

        mock_nlp = MagicMock(return_value=mock_doc)
        extractor._nlp = mock_nlp

        entities = [
            {"name": "Alice", "start": 0, "end": 5},
            {"name": "Google", "start": 12, "end": 18},
        ]
        result = await extractor.extract("Alice works at Google", entities)

        assert len(result) == 1
        assert result[0]["source_name"] == "Alice"
        assert result[0]["target_name"] == "Google"
        assert result[0]["relation"] == "WORKS_AT"
        assert result[0]["confidence"] == 0.8
        assert result[0]["bidirectional"] is False

    @pytest.mark.asyncio
    async def test_prep_relation_lower_confidence(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()

        mock_ent1 = MagicMock()
        mock_ent1.text = "Office"
        mock_ent1.start = 0
        mock_ent1.end = 1
        mock_ent1.root = MagicMock()
        mock_ent1.root.head = MagicMock(pos_="NOUN")

        mock_ent2 = MagicMock()
        mock_ent2.text = "Paris"
        mock_ent2.start = 2
        mock_ent2.end = 3
        mock_ent2.root = MagicMock()
        mock_ent2.root.head = MagicMock(pos_="NOUN")

        prep_tok = MagicMock()
        prep_tok.pos_ = "ADP"
        prep_tok.lemma_ = "in"
        prep_tok.text = "in"

        mock_sent = MagicMock()
        mock_sent.ents = [mock_ent1, mock_ent2]

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_doc.__getitem__ = lambda self, s: (
            [prep_tok] if isinstance(s, slice) else prep_tok
        )
        mock_doc.__len__ = lambda self: 4

        extractor._nlp = MagicMock(return_value=mock_doc)

        result = await extractor.extract(
            "Office in Paris",
            [{"name": "Office"}, {"name": "Paris"}],
        )

        assert len(result) == 1
        assert result[0]["relation"] == "IN"
        assert result[0]["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_no_relation_when_no_verb_or_prep(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()

        mock_ent1 = MagicMock()
        mock_ent1.text = "Alice"
        mock_ent1.start = 0
        mock_ent1.end = 1
        mock_ent1.root = MagicMock()
        mock_ent1.root.head = MagicMock(pos_="NOUN")

        mock_ent2 = MagicMock()
        mock_ent2.text = "Bob"
        mock_ent2.start = 1
        mock_ent2.end = 2
        mock_ent2.root = MagicMock()
        mock_ent2.root.head = MagicMock(pos_="NOUN")

        mock_sent = MagicMock()
        mock_sent.ents = [mock_ent1, mock_ent2]

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_doc.__getitem__ = lambda self, s: []
        mock_doc.__len__ = lambda self: 3

        extractor._nlp = MagicMock(return_value=mock_doc)

        result = await extractor.extract(
            "Alice Bob", [{"name": "Alice"}, {"name": "Bob"}]
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_symmetric_relation(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()

        mock_ent1 = MagicMock()
        mock_ent1.text = "Alice"
        mock_ent1.start = 0
        mock_ent1.end = 1
        mock_ent1.root = MagicMock()
        mock_ent1.root.head = MagicMock(pos_="NOUN")

        mock_ent2 = MagicMock()
        mock_ent2.text = "Bob"
        mock_ent2.start = 3
        mock_ent2.end = 4
        mock_ent2.root = MagicMock()
        mock_ent2.root.head = MagicMock(pos_="NOUN")

        verb_tok = MagicMock()
        verb_tok.pos_ = "VERB"
        verb_tok.lemma_ = "collaborate"
        verb_tok.text = "collaborates"

        mock_sent = MagicMock()
        mock_sent.ents = [mock_ent1, mock_ent2]

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent]
        mock_doc.__getitem__ = lambda self, s: (
            [verb_tok] if isinstance(s, slice) else verb_tok
        )
        mock_doc.__len__ = lambda self: 5

        extractor._nlp = MagicMock(return_value=mock_doc)

        result = await extractor.extract(
            "Alice collaborates with Bob",
            [{"name": "Alice"}, {"name": "Bob"}],
        )
        assert len(result) == 1
        assert result[0]["relation"] == "COLLABORATES_WITH"
        assert result[0]["bidirectional"] is True

    @pytest.mark.asyncio
    async def test_deduplication(self):
        from hermes.relation_extractor import SpacyRelationExtractor

        extractor = SpacyRelationExtractor()

        mock_ent1 = MagicMock()
        mock_ent1.text = "Alice"
        mock_ent1.start = 0
        mock_ent1.end = 1
        mock_ent1.root = MagicMock()
        mock_ent1.root.head = MagicMock(pos_="NOUN")

        mock_ent2 = MagicMock()
        mock_ent2.text = "Google"
        mock_ent2.start = 3
        mock_ent2.end = 4
        mock_ent2.root = MagicMock()
        mock_ent2.root.head = MagicMock(pos_="NOUN")

        verb_tok = MagicMock()
        verb_tok.pos_ = "VERB"
        verb_tok.lemma_ = "work"
        verb_tok.text = "works"

        # Same entities appear in two sentences
        mock_sent1 = MagicMock()
        mock_sent1.ents = [mock_ent1, mock_ent2]
        mock_sent2 = MagicMock()
        mock_sent2.ents = [mock_ent1, mock_ent2]

        mock_doc = MagicMock()
        mock_doc.sents = [mock_sent1, mock_sent2]
        mock_doc.__getitem__ = lambda self, s: (
            [verb_tok] if isinstance(s, slice) else verb_tok
        )
        mock_doc.__len__ = lambda self: 10

        extractor._nlp = MagicMock(return_value=mock_doc)

        result = await extractor.extract(
            "Alice works at Google. Alice works at Google.",
            [{"name": "Alice"}, {"name": "Google"}],
        )
        # Should deduplicate — only one relation
        assert len(result) == 1


class TestRelationExtractorFactory:
    """Tests for get_relation_extractor factory."""

    def test_get_relation_extractor_spacy(self):
        import hermes.relation_extractor as mod

        mod._extractor = None
        with patch.object(mod, "get_env_value", return_value="spacy"):
            extractor = mod.get_relation_extractor()
            assert isinstance(extractor, mod.SpacyRelationExtractor)
        mod._extractor = None

    def test_get_relation_extractor_default(self):
        import hermes.relation_extractor as mod

        mod._extractor = None
        with patch.object(mod, "get_env_value", return_value=None):
            extractor = mod.get_relation_extractor()
            assert isinstance(extractor, mod.SpacyRelationExtractor)
        mod._extractor = None

    def test_get_relation_extractor_unknown_raises(self):
        import hermes.relation_extractor as mod

        mod._extractor = None
        with patch.object(mod, "get_env_value", return_value="unknown"):
            with pytest.raises(ValueError, match="Unknown RELATION_EXTRACTOR"):
                mod.get_relation_extractor()
        mod._extractor = None

    def test_singleton_returns_cached(self):
        import hermes.relation_extractor as mod

        sentinel = MagicMock()
        mod._extractor = sentinel
        assert mod.get_relation_extractor() is sentinel
        mod._extractor = None


class TestVerbToRelation:
    """Tests for verb-to-relation mapping constants."""

    def test_known_verbs(self):
        from hermes.relation_extractor import _VERB_TO_RELATION

        assert _VERB_TO_RELATION["work"] == "WORKS_AT"
        assert _VERB_TO_RELATION["locate"] == "LOCATED_IN"
        assert _VERB_TO_RELATION["develop"] == "DEVELOPS"
        assert _VERB_TO_RELATION["collaborate"] == "COLLABORATES_WITH"

    def test_symmetric_relations(self):
        from hermes.relation_extractor import _SYMMETRIC_RELATIONS

        assert "COLLABORATES_WITH" in _SYMMETRIC_RELATIONS
