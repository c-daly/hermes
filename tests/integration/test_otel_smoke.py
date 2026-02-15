"""Smoke tests for OpenTelemetry instrumentation in Hermes."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from logos_observability import setup_telemetry


@pytest.fixture(autouse=True)
def reset_tracer_provider():
    """Reset the global tracer provider between tests."""
    yield
    # Properly reset global state — set_tracer_provider is guarded by a
    # set-once lock, so we must reset the internal flag to allow re-setting.
    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        trace._TRACER_PROVIDER_SET_ONCE = type(trace._TRACER_PROVIDER_SET_ONCE)()
    if hasattr(trace, "_TRACER_PROVIDER"):
        trace._TRACER_PROVIDER = None


def test_hermes_telemetry_setup():
    """Verify setup_telemetry configures a working TracerProvider."""
    provider = setup_telemetry(service_name="hermes", export_to_console=False)
    assert provider is not None

    tracer = provider.get_tracer("hermes.test")
    with tracer.start_as_current_span("hermes.test_span") as span:
        span.set_attribute("test.key", "test_value")


def test_hermes_spans_have_correct_service_name():
    """Verify spans carry the correct service.name resource attribute."""
    provider = setup_telemetry(service_name="hermes", export_to_console=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("hermes.api")
    with tracer.start_as_current_span("hermes.embed") as span:
        span.set_attribute("embed.model", "test-model")

    provider.force_flush()

    spans = exporter.get_finished_spans()
    assert len(spans) > 0
    assert spans[0].name == "hermes.embed"
    assert spans[0].resource.attributes["service.name"] == "hermes"


def test_hermes_nested_spans():
    """Verify nested spans maintain parent-child relationships."""
    provider = setup_telemetry(service_name="hermes", export_to_console=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = provider.get_tracer("hermes.api")
    with tracer.start_as_current_span("hermes.llm"):
        with tracer.start_as_current_span("hermes.embed") as child:
            child.set_attribute("embed.dimension", 768)

    provider.force_flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    child_span = next(s for s in spans if s.name == "hermes.embed")
    parent_span = next(s for s in spans if s.name == "hermes.llm")
    assert child_span.parent.span_id == parent_span.context.span_id
