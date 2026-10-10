"""OpenTelemetry tracer provider exporting to Arize Phoenix.

Spans are batched and exported in the background. On Cloud Run with request-based
billing the CPU is throttled between requests, so the batch only drains while
requests run; the app therefore flushes on shutdown (Cloud Run sends SIGTERM and
allows a grace period).
"""

import logging
from functools import lru_cache

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from chef_backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


def build_tracer_provider(settings: Settings) -> TracerProvider | None:
    """A provider exporting to Phoenix, or None when tracing is not configured."""
    if not settings.phoenix_endpoint or settings.phoenix_api_key is None:
        if settings.environment == "prod":
            logger.warning("Phoenix tracing is not configured; AI calls are not traced")
        return None

    api_key = settings.phoenix_api_key.get_secret_value()
    exporter = OTLPSpanExporter(
        endpoint=settings.phoenix_endpoint,
        # Phoenix Cloud accepts either header; send both, as the Android app does.
        headers={"Authorization": f"Bearer {api_key}", "api_key": api_key},
    )
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "chef-backend",
                "openinference.project.name": settings.phoenix_project_name,
            }
        )
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    return provider


@lru_cache
def get_tracer_provider() -> TracerProvider | None:
    return build_tracer_provider(get_settings())


def shutdown_tracing() -> None:
    """Export buffered spans before the instance stops."""
    provider = get_tracer_provider()
    if provider is not None:
        provider.shutdown()
