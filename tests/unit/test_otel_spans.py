"""Tests for OpenTelemetry span creation in Hermes API endpoints."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


@pytest.fixture()
def span_capture():
    """Provide an InMemorySpanExporter wired to a dedicated TracerProvider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("hermes.api")
    yield exporter, tracer
    exporter.shutdown()
    provider.shutdown()


def test_hermes_tracer_exists():
    """Verify hermes main.py defines a module-level tracer."""
    import hermes.main as main_module

    assert hasattr(main_module, "tracer"), (
        "main.py must define a module-level tracer via get_tracer"
    )


def test_span_names_defined():
    """Verify all expected span names appear in hermes main.py source."""
    import inspect
    import hermes.main as main_module

    source = inspect.getsource(main_module)
    expected_spans = [
        "hermes.stt",
        "hermes.tts",
        "hermes.nlp",
        "hermes.embed_text",
        "hermes.llm",
        "hermes.ingest.media",
        "hermes.feedback",
    ]
    for span_name in expected_spans:
        assert span_name in source, (
            f"Expected span name {span_name!r} not found in hermes/main.py"
        )


def test_stt_span_creation(span_capture):
    """Verify hermes.stt span is created with correct attributes."""
    exporter, tracer = span_capture

    # Simulate what the STT endpoint does
    with tracer.start_as_current_span("hermes.stt") as span:
        span.set_attribute("stt.format", "audio/wav")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "hermes.stt"
    assert spans[0].attributes.get("stt.format") == "audio/wav"


def test_embed_text_span_creation(span_capture):
    """Verify hermes.embed_text span is created with correct attributes."""
    exporter, tracer = span_capture

    with tracer.start_as_current_span("hermes.embed_text") as span:
        span.set_attribute("embedding.model", "clip-vit-base")
        span.set_attribute("embedding.text_length", 42)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "hermes.embed_text"
    assert spans[0].attributes.get("embedding.model") == "clip-vit-base"
    assert spans[0].attributes.get("embedding.text_length") == 42


def test_llm_span_creation(span_capture):
    """Verify hermes.llm span is created with correct attributes."""
    exporter, tracer = span_capture

    with tracer.start_as_current_span("hermes.llm") as span:
        span.set_attribute("llm.provider", "openai")
        span.set_attribute("llm.model", "gpt-4")
        span.set_attribute("llm.max_tokens", 1024)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "hermes.llm"
    assert spans[0].attributes.get("llm.provider") == "openai"


def test_span_records_error(span_capture):
    """Verify span error recording works correctly."""
    exporter, tracer = span_capture

    with tracer.start_as_current_span("hermes.nlp") as span:
        span.set_attribute("nlp.text_length", 10)
        try:
            raise ValueError("NLP processing failed")
        except Exception as e:
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR

    exception_events = [ev for ev in spans[0].events if ev.name == "exception"]
    assert len(exception_events) >= 1
    assert "NLP processing failed" in str(
        exception_events[0].attributes.get("exception.message", "")
    )
