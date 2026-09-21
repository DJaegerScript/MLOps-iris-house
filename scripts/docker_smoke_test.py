#!/usr/bin/env python3
"""Build and smoke-test the application-only Iris Docker image.

The default mode verifies the image, exposed port, and Streamlit health endpoint
without requiring a model to be present. ``--sample-prediction`` is intentionally
stricter: it requires the real S3 object coordinates and uses Playwright against
the running container, so a prediction can never be faked by a local fixture.
AWS credentials are never read from or copied into the image. For local testing,
``--aws-profile`` mounts an existing read-only AWS config directory at runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

CONTAINER_PORT = 8501
HEALTH_PATH = "/_stcore/health"
S3_CONFIG_KEYS = (
    "IRIS_MODEL_S3_BUCKET",
    "IRIS_MODEL_S3_KEY",
    "IRIS_MODEL_S3_VERSION_ID",
)
APP_CONFIG_DEFAULTS = {
    "IRIS_MODEL_S3_BUCKET": "docker-smoke-placeholder",
    "IRIS_MODEL_S3_KEY": "docker-smoke-placeholder",
    "IRIS_MODEL_S3_VERSION_ID": "docker-smoke-placeholder",
    "IRIS_MODEL_VERSION": "v1",
    "IRIS_ENVIRONMENT": "docker-smoke",
    "GIT_COMMIT_SHA": "docker-smoke",
    "DOCKER_IMAGE_VERSION": "docker-smoke",
    "LOW_CONFIDENCE_THRESHOLD": "0.75",
}


class SmokeConfigurationError(ValueError):
    """Raised when a smoke mode is missing required runtime configuration."""


def validate_live_config(environment: Mapping[str, str | None]) -> None:
    """Require all S3 coordinates before allowing a live prediction."""

    missing = []
    for key in S3_CONFIG_KEYS:
        value = environment.get(key)
        if value is None or not str(value).strip():
            missing.append(key)
    if missing:
        raise SmokeConfigurationError(
            "S3 model configuration is required for --sample-prediction: "
            + ", ".join(missing)
        )


def _run(
    command: Sequence[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        capture_output=capture_output,
    )


def build_image(image: str, context: Path) -> None:
    """Build the requested image from the repository context."""

    _run(("docker", "build", "--tag", image, str(context)))


def assert_exposed_port(image: str) -> None:
    """Verify the image metadata exposes the Streamlit port."""

    result = _run(
        (
            "docker",
            "image",
            "inspect",
            "--format",
            "{{json .Config.ExposedPorts}}",
            image,
        ),
        capture_output=True,
    )
    try:
        exposed_ports = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "docker image inspect returned invalid port metadata"
        ) from error
    if not isinstance(exposed_ports, dict) or "8501/tcp" not in exposed_ports:
        raise RuntimeError("Docker image does not expose 8501/tcp")


def _runtime_environment(args: argparse.Namespace) -> dict[str, str]:
    values = dict(APP_CONFIG_DEFAULTS)
    for argument, variable in (
        ("model_s3_bucket", "IRIS_MODEL_S3_BUCKET"),
        ("model_s3_key", "IRIS_MODEL_S3_KEY"),
        ("model_s3_version_id", "IRIS_MODEL_S3_VERSION_ID"),
        ("model_version", "IRIS_MODEL_VERSION"),
        ("environment", "IRIS_ENVIRONMENT"),
        ("git_commit_sha", "GIT_COMMIT_SHA"),
        ("docker_image_version", "DOCKER_IMAGE_VERSION"),
        ("low_confidence_threshold", "LOW_CONFIDENCE_THRESHOLD"),
    ):
        value = getattr(args, argument)
        if value is not None:
            values[variable] = str(value)
    return values


def start_container(
    image: str,
    host_port: int,
    environment: Mapping[str, str],
    *,
    aws_profile: str | None = None,
    aws_config_dir: Path | None = None,
) -> str:
    """Start a locked-down container and return its Docker container ID."""

    command = [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--publish",
        f"{host_port}:{CONTAINER_PORT}",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
    ]
    for key, value in environment.items():
        command.extend(("--env", f"{key}={value}"))

    if aws_profile is not None:
        profile_dir = (
            aws_config_dir
            if aws_config_dir is not None
            else Path(os.environ.get("AWS_CONFIG_DIR", Path.home() / ".aws"))
        ).expanduser()
        if not profile_dir.is_dir():
            raise SmokeConfigurationError(
                f"AWS config directory does not exist: {profile_dir}"
            )
        command.extend(
            (
                "--env",
                f"AWS_PROFILE={aws_profile}",
                "--env",
                "AWS_SDK_LOAD_CONFIG=1",
                "--mount",
                f"type=bind,source={profile_dir},destination=/home/iris/.aws,readonly",
            )
        )

    command.append(image)
    result = _run(command, capture_output=True)
    container_id = result.stdout.strip()
    if not container_id:
        raise RuntimeError("docker run did not return a container ID")
    return container_id


def wait_for_health(base_url: str, timeout_seconds: float) -> None:
    """Wait until Streamlit's health endpoint returns HTTP 200."""

    deadline = time.monotonic() + timeout_seconds
    url = f"{base_url.rstrip('/')}{HEALTH_PATH}"
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                body = response.read().decode("utf-8", errors="replace").strip()
                if response.status == 200 and body == "ok":
                    return
                last_error = f"HTTP {response.status}: {body!r}"
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(1)
    raise RuntimeError(f"Streamlit health check failed: {last_error}")


def submit_sample_prediction(base_url: str, timeout_seconds: float) -> None:
    """Submit representative setosa inputs through the real Streamlit UI."""

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise SmokeConfigurationError(
            "--sample-prediction requires Playwright; install it and Chromium first"
        ) from error

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(
                base_url,
                wait_until="domcontentloaded",
                timeout=int(timeout_seconds * 1000),
            )
            for label, value in (
                ("Sepal Length", "5.1"),
                ("Sepal Width", "3.5"),
                ("Petal Length", "1.4"),
                ("Petal Width", "0.2"),
            ):
                page.get_by_label(label).fill(value)
            page.get_by_role("button", name="Predict species").click()
            page.get_by_text("Predicted species:", exact=False).wait_for(
                state="visible", timeout=int(timeout_seconds * 1000)
            )
        finally:
            browser.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="iris-mlops:smoke")
    parser.add_argument("--context", type=Path, default=Path("."))
    parser.add_argument("--host-port", type=int, default=8501)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--sample-prediction", action="store_true")
    parser.add_argument("--aws-profile")
    parser.add_argument("--aws-config-dir", type=Path)
    parser.add_argument("--model-s3-bucket")
    parser.add_argument("--model-s3-key")
    parser.add_argument("--model-s3-version-id")
    parser.add_argument("--model-version")
    parser.add_argument("--environment")
    parser.add_argument("--git-commit-sha")
    parser.add_argument("--docker-image-version")
    parser.add_argument("--low-confidence-threshold")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    environment = _runtime_environment(args)
    explicit_s3_values = {
        "IRIS_MODEL_S3_BUCKET": args.model_s3_bucket,
        "IRIS_MODEL_S3_KEY": args.model_s3_key,
        "IRIS_MODEL_S3_VERSION_ID": args.model_s3_version_id,
    }
    if any(value is not None for value in explicit_s3_values.values()):
        validate_live_config(explicit_s3_values)
    if args.sample_prediction:
        validate_live_config(environment)

    context = args.context.resolve()
    if not args.no_build:
        build_image(args.image, context)
    assert_exposed_port(args.image)

    container_id = start_container(
        args.image,
        args.host_port,
        environment,
        aws_profile=args.aws_profile,
        aws_config_dir=args.aws_config_dir,
    )
    base_url = f"http://127.0.0.1:{args.host_port}"
    try:
        wait_for_health(base_url, args.timeout_seconds)
        if args.sample_prediction:
            submit_sample_prediction(base_url, args.timeout_seconds)
            print("Docker smoke test passed with a live S3-backed sample prediction.")
        else:
            print("Docker smoke test passed health-only checks.")
    finally:
        subprocess.run(
            ("docker", "rm", "--force", container_id),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        SmokeConfigurationError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Docker smoke test failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
