"""Process-level Langfuse client, callback, and lifecycle helpers."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from langfuse import Langfuse, propagate_attributes
from langfuse.api import TraceBody
from langfuse.langchain import CallbackHandler
from opentelemetry.sdk.trace import TracerProvider

from .langfuse_settings import LangfuseSettings


class LangfuseTraceRun:
    """Holds the final output for one Langfuse trace run."""

    def __init__(self) -> None:
        self.output: Any | None = None

    def set_output(self, output: Any) -> None:
        self.output = output


class LangfuseObservability:
    """Configures Langfuse and enriches each LangGraph run."""

    def __init__(
        self,
        settings: LangfuseSettings,
        *,
        tracer_provider: TracerProvider | None = None,
    ) -> None:
        settings.validate()
        self._settings = settings
        self._client: Langfuse | None = None
        self._handler: CallbackHandler | None = None

        if not settings.enabled:
            return

        if settings.bypass_proxy:
            _bypass_proxy_for(settings.base_url or "")

        self._client = Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            debug=settings.debug,
            flush_at=settings.flush_at,
            flush_interval=settings.flush_interval,
            environment=settings.environment,
            release=settings.release,
            sample_rate=settings.sample_rate,
            tracer_provider=tracer_provider,
        )
        self._handler = CallbackHandler(public_key=settings.public_key)

    @property
    def enabled(self) -> bool:
        return self._handler is not None

    @property
    def base_url(self) -> str | None:
        return self._settings.base_url

    @property
    def instance_ip(self) -> str | None:
        return self._settings.instance_ip

    def configure_run(
        self,
        config: dict[str, Any],
        *,
        run_name: str,
        session_id: str,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._handler is None:
            return config

        configured = dict(config)
        callbacks = list(configured.get("callbacks") or [])
        callbacks.append(self._handler)
        configured["callbacks"] = callbacks
        configured["run_name"] = run_name

        run_metadata = dict(configured.get("metadata") or {})
        if metadata:
            run_metadata.update(metadata)
        run_metadata["langfuse_session_id"] = session_id
        run_metadata["langfuse_trace_name"] = run_name
        run_metadata["langfuse_tags"] = list(
            dict.fromkeys((*self._settings.tags, *tags))
        )
        configured["metadata"] = run_metadata
        return configured

    @contextmanager
    def trace_run(
        self,
        *,
        run_name: str,
        session_id: str,
        input: Any,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[LangfuseTraceRun]:
        trace_run = LangfuseTraceRun()
        if self._client is None:
            yield trace_run
            return

        merged_tags = list(dict.fromkeys((*self._settings.tags, *tags)))
        with propagate_attributes(
            session_id=session_id,
            trace_name=run_name,
            tags=merged_tags,
            metadata=metadata,
        ):
            trace_input = self._prepare_trace_input(input)
            with self._client.start_as_current_observation(
                name=run_name,
                as_type="agent",
                input=trace_input,
                metadata=metadata,
            ) as observation:
                self._enqueue_trace_input_update(input=trace_input)
                try:
                    yield trace_run
                finally:
                    observation.update(output=trace_run.output)

    @staticmethod
    def _prepare_trace_input(value: Any) -> Any:
        if not isinstance(value, str):
            return value

        text = value.strip()
        if len(text) < 2 or text[0] != '"' or text[-1] != '"':
            return value

        try:
            decoded = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value

        return decoded if isinstance(decoded, str) else value

    def _enqueue_trace_input_update(self, *, input: Any) -> None:
        if self._client is None:
            return

        trace_id = self._client.get_current_trace_id()
        resources = getattr(self._client, "_resources", None)
        if not trace_id or resources is None:
            return

        event = {
            "id": self._client.create_trace_id(),
            "type": "trace-create",
            "timestamp": datetime.now(UTC),
            "body": TraceBody(id=trace_id, input=input),
        }
        resources.add_trace_task(event)

    def flush(self) -> None:
        if self._client is not None:
            self._client.flush()

    def shutdown(self) -> None:
        if self._client is not None:
            self._client.shutdown()


def _bypass_proxy_for(base_url: str) -> None:
    hostname = urlparse(base_url).hostname
    if not hostname:
        return
    for variable in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(variable, "")
        entries = [entry.strip() for entry in existing.split(",") if entry.strip()]
        if hostname not in entries:
            os.environ[variable] = ",".join((hostname, *entries))
