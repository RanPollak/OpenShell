# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for explicit inference routing via `inference.local`.

In the new model, sandbox traffic is routed only when the request targets
`inference.local`. There is no implicit catch-all interception for arbitrary
hosts like `api.openai.com`.
"""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from typing import TYPE_CHECKING

import grpc
import pytest

from openshell._proto import datamodel_pb2, openshell_pb2, sandbox_pb2

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from openshell import (
        ClusterInferenceConfig,
        InferenceRouteClient,
        Sandbox,
        SandboxClient,
    )


_BASE_FILESYSTEM = sandbox_pb2.FilesystemPolicy(
    include_workdir=True,
    read_only=["/usr", "/lib", "/etc", "/app", "/var/log", "/proc", "/dev/urandom"],
    read_write=["/sandbox", "/tmp"],
)
_BASE_LANDLOCK = sandbox_pb2.LandlockPolicy(compatibility="best_effort")
_BASE_PROCESS = sandbox_pb2.ProcessPolicy(run_as_user="sandbox", run_as_group="sandbox")

pytestmark = pytest.mark.xdist_group("inference-routing")

_MANAGED_OPENAI_MODEL_ID = "mock/e2e-openai-model"
_MANAGED_OPENAI_PROVIDER_NAME = "e2e-managed-openai"
_INFERENCE_CONFIG_LOCK = "/tmp/openshell-e2e-inference-config.lock"


def _baseline_policy() -> sandbox_pb2.SandboxPolicy:
    return sandbox_pb2.SandboxPolicy(
        version=1,
        filesystem=_BASE_FILESYSTEM,
        landlock=_BASE_LANDLOCK,
        process=_BASE_PROCESS,
    )


def _upsert_managed_inference(
    inference_client: InferenceRouteClient,
    sandbox_client: SandboxClient,
    *,
    provider_name: str,
    provider_type: str,
    credential_key: str,
    base_url_key: str,
    model_id: str,
    base_url: str,
    model_source: str = "",
) -> None:
    provider = datamodel_pb2.Provider(
        metadata=datamodel_pb2.ObjectMeta(name=provider_name),
        type=provider_type,
        credentials={credential_key: "mock"},
        config={
            base_url_key: base_url,
        },
    )
    timeout = sandbox_client._timeout

    for _ in range(5):
        try:
            sandbox_client._stub.UpdateProvider(
                openshell_pb2.UpdateProviderRequest(provider=provider),
                timeout=timeout,
            )
            break
        except grpc.RpcError as exc:
            if exc.code() != grpc.StatusCode.NOT_FOUND:
                raise

            try:
                sandbox_client._stub.CreateProvider(
                    openshell_pb2.CreateProviderRequest(provider=provider),
                    timeout=timeout,
                )
                break
            except grpc.RpcError as create_exc:
                if create_exc.code() == grpc.StatusCode.ALREADY_EXISTS:
                    continue
                raise
    else:
        raise RuntimeError("failed to upsert managed e2e provider after retries")

    inference_client.set_cluster(
        provider_name=provider_name,
        model_id=model_id,
        model_source=model_source,
    )


def _current_cluster_inference(
    inference_client: InferenceRouteClient,
) -> ClusterInferenceConfig | None:
    try:
        return inference_client.get_cluster()
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.NOT_FOUND:
            return None
        raise


def _restore_cluster_inference(
    inference_client: InferenceRouteClient,
    previous: ClusterInferenceConfig | None,
) -> None:
    if previous is None:
        return

    inference_client.set_cluster(
        provider_name=previous.provider_name,
        model_id=previous.model_id,
        model_source=previous.model_source,
        # Teardown restores prior shared state as-is, even if the previous
        # route is intentionally unreachable or no longer verifiable.
        no_verify=True,
    )


@contextmanager
def _cluster_config_lock() -> Iterator[None]:
    with open(_INFERENCE_CONFIG_LOCK, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@pytest.fixture
def managed_openai_route(
    inference_client: InferenceRouteClient,
    sandbox_client: SandboxClient,
) -> Iterator[str]:
    with _cluster_config_lock():
        previous = _current_cluster_inference(inference_client)
        _upsert_managed_inference(
            inference_client,
            sandbox_client,
            provider_name=_MANAGED_OPENAI_PROVIDER_NAME,
            provider_type="openai",
            credential_key="OPENAI_API_KEY",
            base_url_key="OPENAI_BASE_URL",
            model_id=_MANAGED_OPENAI_MODEL_ID,
            base_url="mock://e2e-managed-openai",
        )
        try:
            yield _MANAGED_OPENAI_MODEL_ID
        finally:
            _restore_cluster_inference(inference_client, previous)


def test_model_discovery_call_routed_to_backend(
    sandbox: Callable[..., Sandbox],
    managed_openai_route: str,
) -> None:
    """Model discovery endpoint is treated as an inference protocol."""
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())

    def call_models() -> str:
        import ssl
        import urllib.request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request("https://inference.local/v1/models", method="GET")
        resp = urllib.request.urlopen(req, timeout=30, context=ctx)
        return resp.read().decode()

    with sandbox(spec=spec, delete_on_exit=True) as sb:
        result = sb.exec_python(call_models, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        output = result.stdout.strip()
        assert "Hello from openshell mock backend" in output
        assert managed_openai_route in output


def test_inference_call_routed_to_backend(
    sandbox: Callable[..., Sandbox],
    managed_openai_route: str,
) -> None:
    """OpenAI chat request to `inference.local` is intercepted and routed."""
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())

    def call_chat_completions() -> str:
        import json
        import ssl
        import urllib.request

        body = json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode()

        req = urllib.request.Request(
            "https://inference.local/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy-key",
            },
            method="POST",
        )
        # The proxy will TLS-terminate, so we need to accept its cert.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        resp = urllib.request.urlopen(req, timeout=30, context=ctx)
        return resp.read().decode()

    with sandbox(spec=spec, delete_on_exit=True) as sb:
        result = sb.exec_python(call_chat_completions, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        output = result.stdout.strip()
        assert "Hello from openshell mock backend" in output
        assert managed_openai_route in output


def test_non_inference_request_denied(
    sandbox: Callable[..., Sandbox],
    managed_openai_route: str,
) -> None:
    """Non-inference path on `inference.local` is denied with 403."""
    _ = managed_openai_route
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())

    def make_non_inference_request() -> str:
        import ssl
        import urllib.error
        import urllib.request

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            req = urllib.request.Request("https://inference.local/v1/not-inference")
            urllib.request.urlopen(req, timeout=10, context=ctx)
            return "unexpected_success"
        except urllib.error.HTTPError as e:
            return f"http_error_{e.code}"
        except Exception as e:
            return f"error: {e}"

    with sandbox(spec=spec, delete_on_exit=True) as sb:
        result = sb.exec_python(make_non_inference_request, timeout_seconds=30)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        assert result.stdout.strip() == "http_error_403"


def test_unsupported_protocol_returns_400(
    sandbox: Callable[..., Sandbox],
    managed_openai_route: str,
) -> None:
    """Protocol mismatch returns 400 when no compatible route exists."""
    _ = managed_openai_route
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())

    def call_anthropic_messages() -> str:
        import json
        import ssl
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": "claude-test",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode()

        req = urllib.request.Request(
            "https://inference.local/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": "dummy-key",
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            resp = urllib.request.urlopen(req, timeout=30, context=ctx)
            return resp.read().decode()
        except urllib.error.HTTPError as exc:
            return (
                f"http_error_{exc.code}:{exc.read().decode('utf-8', errors='replace')}"
            )

    with sandbox(spec=spec, delete_on_exit=True) as sb:
        result = sb.exec_python(call_anthropic_messages, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        output = result.stdout.strip()
        assert output.startswith("http_error_400"), output
        assert "no compatible inference route" in output


def test_non_inference_host_is_not_intercepted(
    sandbox: Callable[..., Sandbox],
    managed_openai_route: str,
) -> None:
    """Requests to non-`inference.local` hosts do not get inference routing."""
    _ = managed_openai_route
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())

    def call_external_openai_endpoint() -> str:
        import json
        import ssl
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode()

        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            resp = urllib.request.urlopen(req, timeout=30, context=ctx)
            return resp.read().decode()
        except urllib.error.URLError as exc:
            return f"url_error:{exc}"
        except Exception as exc:
            return f"error:{type(exc).__name__}:{exc}"

    with sandbox(spec=spec, delete_on_exit=True) as sb:
        result = sb.exec_python(call_external_openai_endpoint, timeout_seconds=60)
        assert result.exit_code == 0, f"stderr: {result.stderr}"
        output = result.stdout.strip()
        assert "Tunnel connection failed: 403 Forbidden" in output


# ---------------------------------------------------------------------------
# Model-source policy tests
#
# These exercise the per-route policy that decides how the privacy router
# treats the client-supplied `model` field. The strongest end-to-end signal
# comes from the `matching` policy: a mismatched model produces a 400
# response from the router *before* the mock backend is touched, so the
# test can rely on the HTTP status without coupling to the mock body.
# ---------------------------------------------------------------------------


_MODEL_SOURCE_MATCHING_MODEL_ID = "mock/model-source-test"
_MODEL_SOURCE_MATCHING_PROVIDER_NAME = "e2e-model-source-openai"


def _exec_chat_request(sb: Sandbox, *, model: str) -> tuple[int, str]:
    """Send one `/v1/chat/completions` request from inside the sandbox.

    Returns `(http_status, body)`. The body for 400 responses includes the
    router's `InvalidRequest` detail, which the matching-mode tests assert on.
    """

    def call(model: str = model) -> str:
        import json
        import ssl
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
            }
        ).encode()

        req = urllib.request.Request(
            "https://inference.local/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummy-key",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            resp = urllib.request.urlopen(req, timeout=30, context=ctx)
            return f"{resp.status}|{resp.read().decode()}"
        except urllib.error.HTTPError as exc:
            return f"{exc.code}|{exc.read().decode('utf-8', errors='replace')}"

    result = sb.exec_python(call, timeout_seconds=60)
    assert result.exit_code == 0, f"stderr: {result.stderr}"
    status_str, _, body = result.stdout.strip().partition("|")
    return int(status_str), body


@pytest.fixture
def matching_mode_route(
    inference_client: InferenceRouteClient,
    sandbox_client: SandboxClient,
) -> Iterator[str]:
    """Persist a route configured with `model_source=matching` for the test."""
    with _cluster_config_lock():
        previous = _current_cluster_inference(inference_client)
        _upsert_managed_inference(
            inference_client,
            sandbox_client,
            provider_name=_MODEL_SOURCE_MATCHING_PROVIDER_NAME,
            provider_type="openai",
            credential_key="OPENAI_API_KEY",
            base_url_key="OPENAI_BASE_URL",
            model_id=_MODEL_SOURCE_MATCHING_MODEL_ID,
            base_url="mock://e2e-model-source-openai",
            model_source="matching",
        )
        try:
            yield _MODEL_SOURCE_MATCHING_MODEL_ID
        finally:
            _restore_cluster_inference(inference_client, previous)


def test_matching_mode_accepts_client_model_equal_to_route(
    sandbox: Callable[..., Sandbox],
    matching_mode_route: str,
) -> None:
    """`matching` mode lets through a request whose `model` equals `route.model`."""
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())
    with sandbox(spec=spec, delete_on_exit=True) as sb:
        status, body = _exec_chat_request(sb, model=matching_mode_route)
        assert status == 200, f"expected 200, got {status} body={body}"
        assert "Hello from openshell mock backend" in body


def test_matching_mode_rejects_client_model_mismatch(
    sandbox: Callable[..., Sandbox],
    matching_mode_route: str,
) -> None:
    """`matching` mode rejects a mismatched `model` with 400 before reaching the backend."""
    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())
    with sandbox(spec=spec, delete_on_exit=True) as sb:
        status, body = _exec_chat_request(sb, model="this-model-does-not-exist")
        assert status == 400, f"expected 400, got {status} body={body}"
        assert "matching" in body
        assert "this-model-does-not-exist" in body
        assert matching_mode_route in body


def test_model_source_policy_change_invalidates_sandbox_route_cache(
    sandbox: Callable[..., Sandbox],
    inference_client: InferenceRouteClient,
    sandbox_client: SandboxClient,
) -> None:
    """A policy-only update propagates to running sandboxes within the refresh window.

    Regression test for the revision-hash fix: changing only `model_source`
    must bump the bundle revision, otherwise the sandbox-side route cache
    skips the update and the new policy never takes effect.
    """
    refresh_wait_seconds = 7  # DEFAULT_ROUTE_REFRESH_INTERVAL_SECS (5) + slack.

    spec = datamodel_pb2.SandboxSpec(policy=_baseline_policy())
    with _cluster_config_lock():
        previous = _current_cluster_inference(inference_client)
        try:
            _upsert_managed_inference(
                inference_client,
                sandbox_client,
                provider_name=_MODEL_SOURCE_MATCHING_PROVIDER_NAME,
                provider_type="openai",
                credential_key="OPENAI_API_KEY",
                base_url_key="OPENAI_BASE_URL",
                model_id=_MODEL_SOURCE_MATCHING_MODEL_ID,
                base_url="mock://e2e-model-source-openai",
                model_source="matching",
            )
            with sandbox(spec=spec, delete_on_exit=True) as sb:
                # Sanity: matching mode rejects the bogus model.
                status, _ = _exec_chat_request(sb, model="bogus-model")
                assert status == 400, "matching mode should reject upfront"

                # Flip policy only (provider + model unchanged) to caller.
                inference_client.set_cluster(
                    provider_name=_MODEL_SOURCE_MATCHING_PROVIDER_NAME,
                    model_id=_MODEL_SOURCE_MATCHING_MODEL_ID,
                    model_source="caller",
                    no_verify=True,
                )
                # Wait for the sandbox to pick up the new bundle revision.
                import time

                time.sleep(refresh_wait_seconds)

                # Caller mode forwards the bogus model to the backend, which
                # the mock answers with 200 — the matching gate is gone.
                status, body = _exec_chat_request(sb, model="bogus-model")
                assert status == 200, (
                    f"caller mode should reach the mock backend; got {status} body={body}"
                )
        finally:
            _restore_cluster_inference(inference_client, previous)
