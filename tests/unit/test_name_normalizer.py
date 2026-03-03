"""Tests for name_normalizer — lowercasing, singularization, and dedup."""

from hermes.name_normalizer import normalize_entities


class TestLowercasing:
    def test_uppercase_name_lowercased(self):
        entities = [{"name": "Rottweiler", "type": "entity", "start": 0, "end": 10}]
        result = normalize_entities(entities, "Rottweiler")
        assert result[0]["name"] == "rottweiler"

    def test_already_lowercase_unchanged(self):
        entities = [{"name": "dog", "type": "entity", "start": 0, "end": 3}]
        result = normalize_entities(entities, "dog")
        assert result[0]["name"] == "dog"

    def test_mixed_case_lowercased(self):
        entities = [{"name": "EiffelTower", "type": "entity", "start": 0, "end": 11}]
        result = normalize_entities(entities, "EiffelTower")
        assert result[0]["name"] == "eiffeltower"


class TestSingularization:
    def test_simple_s_plural(self):
        entities = [{"name": "dogs", "type": "entity", "start": 0, "end": 4}]
        result = normalize_entities(entities, "dogs")
        assert result[0]["name"] == "dog"

    def test_ies_plural(self):
        entities = [{"name": "Rotties", "type": "entity", "start": 0, "end": 7}]
        result = normalize_entities(entities, "Rotties")
        assert result[0]["name"] == "rottie"

    def test_es_plural(self):
        entities = [{"name": "checkups", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "checkups")
        assert result[0]["name"] == "checkup"

    def test_ches_plural(self):
        entities = [{"name": "watches", "type": "entity", "start": 0, "end": 7}]
        result = normalize_entities(entities, "watches")
        assert result[0]["name"] == "watch"

    def test_shes_plural(self):
        entities = [{"name": "bushes", "type": "entity", "start": 0, "end": 6}]
        result = normalize_entities(entities, "bushes")
        assert result[0]["name"] == "bush"

    def test_xes_plural(self):
        entities = [{"name": "boxes", "type": "entity", "start": 0, "end": 5}]
        result = normalize_entities(entities, "boxes")
        assert result[0]["name"] == "box"

    def test_sses_plural(self):
        entities = [{"name": "grasses", "type": "entity", "start": 0, "end": 7}]
        result = normalize_entities(entities, "grasses")
        assert result[0]["name"] == "grass"

    def test_should_not_singularize_short_words(self):
        """Words <= 2 chars should not be singularized."""
        entities = [{"name": "us", "type": "entity", "start": 0, "end": 2}]
        result = normalize_entities(entities, "us")
        assert result[0]["name"] == "us"

    def test_should_not_singularize_non_plural_s(self):
        """Words like 'paris', 'diabetes' should not be singularized."""
        entities = [{"name": "Paris", "type": "location", "start": 0, "end": 5}]
        result = normalize_entities(entities, "Paris")
        assert result[0]["name"] == "paris"
        assert result[0]["type"] == "location"

    def test_diabetes_not_singularized(self):
        entities = [{"name": "diabetes", "type": "concept", "start": 0, "end": 8}]
        result = normalize_entities(entities, "diabetes")
        assert result[0]["name"] == "diabetes"

    def test_ss_ending_not_singularized(self):
        """Words ending in 'ss' (like 'grass', 'lass') should not lose the s."""
        entities = [{"name": "grass", "type": "entity", "start": 0, "end": 5}]
        result = normalize_entities(entities, "grass")
        assert result[0]["name"] == "grass"


class TestDeduplication:
    def test_duplicate_entities_merged(self):
        entities = [
            {"name": "Rotties", "type": "entity", "start": 10, "end": 17},
            {"name": "rottie", "type": "entity", "start": 30, "end": 36},
        ]
        result = normalize_entities(entities, "I love my Rotties and my rottie")
        assert len(result) == 1
        assert result[0]["name"] == "rottie"

    def test_dedup_keeps_longer_span(self):
        entities = [
            {"name": "dog", "type": "entity", "start": 0, "end": 3},
            {"name": "Dogs", "type": "entity", "start": 10, "end": 14},
        ]
        result = normalize_entities(entities, "dog likes Dogs too")
        assert len(result) == 1
        # "Dogs" has a longer span (4 chars) vs "dog" (3 chars)
        assert result[0]["start"] == 10
        assert result[0]["end"] == 14

    def test_dedup_preserves_type_from_kept_entity(self):
        entities = [
            {"name": "paris", "type": "entity", "start": 0, "end": 5},
            {"name": "Paris", "type": "location", "start": 20, "end": 25},
        ]
        result = normalize_entities(entities, "paris is great, see Paris now")
        assert len(result) == 1
        # "Paris" has a longer span (5 chars same as "paris") — keeps first when equal
        assert result[0]["type"] == "entity"

    def test_no_dedup_different_names(self):
        entities = [
            {"name": "dog", "type": "entity", "start": 0, "end": 3},
            {"name": "cat", "type": "entity", "start": 8, "end": 11},
        ]
        result = normalize_entities(entities, "dog and cat")
        assert len(result) == 2


class TestPreservation:
    def test_preserves_type(self):
        entities = [{"name": "Paris", "type": "location", "start": 0, "end": 5}]
        result = normalize_entities(entities, "Paris")
        assert result[0]["type"] == "location"

    def test_preserves_offsets(self):
        entities = [{"name": "dog", "type": "entity", "start": 4, "end": 7}]
        result = normalize_entities(entities, "the dog ran")
        assert result[0]["start"] == 4
        assert result[0]["end"] == 7


class TestEdgeCases:
    def test_single_char_name(self):
        entities = [{"name": "X", "type": "entity", "start": 0, "end": 1}]
        result = normalize_entities(entities, "X")
        assert result[0]["name"] == "x"

    def test_empty_entities_list(self):
        result = normalize_entities([], "some text")
        assert result == []

    def test_empty_name(self):
        entities = [{"name": "", "type": "entity", "start": 0, "end": 0}]
        result = normalize_entities(entities, "text")
        assert result == []


class TestWordNetLemmatization:
    """Tests for irregular plurals handled by WordNet (not suffix rules)."""

    def test_irregular_children(self):
        entities = [{"name": "Children", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "Children")
        assert result[0]["name"] == "child"

    def test_irregular_mice(self):
        entities = [{"name": "mice", "type": "entity", "start": 0, "end": 4}]
        result = normalize_entities(entities, "mice")
        assert result[0]["name"] == "mouse"

    def test_irregular_people(self):
        entities = [{"name": "People", "type": "entity", "start": 0, "end": 6}]
        result = normalize_entities(entities, "People")
        assert result[0]["name"] == "people"

    def test_irregular_wolves(self):
        entities = [{"name": "wolves", "type": "entity", "start": 0, "end": 6}]
        result = normalize_entities(entities, "wolves")
        assert result[0]["name"] == "wolf"

    def test_irregular_knives(self):
        entities = [{"name": "knives", "type": "entity", "start": 0, "end": 6}]
        result = normalize_entities(entities, "knives")
        assert result[0]["name"] == "knife"

    def test_ies_to_y_companies(self):
        """WordNet correctly maps -ies -> -y for dictionary words."""
        entities = [{"name": "Companies", "type": "entity", "start": 0, "end": 9}]
        result = normalize_entities(entities, "Companies")
        assert result[0]["name"] == "company"

    def test_ies_to_y_entities(self):
        entities = [{"name": "entities", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "entities")
        assert result[0]["name"] == "entity"

    def test_ies_to_y_communities(self):
        entities = [{"name": "communities", "type": "entity", "start": 0, "end": 11}]
        result = normalize_entities(entities, "communities")
        assert result[0]["name"] == "community"

    def test_latin_plural_matrices(self):
        entities = [{"name": "matrices", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "matrices")
        assert result[0]["name"] == "matrix"

    def test_latin_plural_analyses(self):
        entities = [{"name": "analyses", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "analyses")
        assert result[0]["name"] == "analysis"

    def test_buses(self):
        entities = [{"name": "buses", "type": "entity", "start": 0, "end": 5}]
        result = normalize_entities(entities, "buses")
        assert result[0]["name"] == "bus"

    def test_heroes(self):
        entities = [{"name": "heroes", "type": "entity", "start": 0, "end": 6}]
        result = normalize_entities(entities, "heroes")
        assert result[0]["name"] == "hero"

    def test_potatoes(self):
        entities = [{"name": "potatoes", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "potatoes")
        assert result[0]["name"] == "potato"

    def test_multi_word_entity(self):
        """Each word in a multi-word name is lemmatized independently."""
        entities = [{"name": "dog breeds", "type": "entity", "start": 0, "end": 10}]
        result = normalize_entities(entities, "dog breeds")
        assert result[0]["name"] == "dog breed"

    def test_multi_word_proper_noun_preserved(self):
        """Capitalized multi-word entities are proper nouns — no lemmatization."""
        entities = [{"name": "Field Mice", "type": "entity", "start": 0, "end": 10}]
        result = normalize_entities(entities, "Field Mice")
        assert result[0]["name"] == "field mice"

    def test_multi_word_lowercase_lemmatized(self):
        """Lowercase multi-word entities are lemmatized normally."""
        entities = [{"name": "field mice", "type": "entity", "start": 0, "end": 10}]
        result = normalize_entities(entities, "field mice")
        assert result[0]["name"] == "field mouse"

    def test_possessive_stripped(self):
        entities = [{"name": "rottie's", "type": "entity", "start": 0, "end": 8}]
        result = normalize_entities(entities, "rottie's")
        assert result[0]["name"] == "rottie"

    def test_proper_noun_possessive(self):
        entities = [{"name": "United States'", "type": "entity", "start": 0, "end": 14}]
        result = normalize_entities(entities, "United States'")
        assert result[0]["name"] == "united states"
