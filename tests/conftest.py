"""Keep the test suite hermetic: tests may only talk to the Firestore emulator.

This is an allowlist, not a blocklist of known-bad settings:

- Google credentials are never available. Every Google client library resolves
  Application Default Credentials through google.auth.default, which is replaced
  here before any test module is imported. A developer's ADC (gcloud login) can
  therefore never be picked up, whatever project a test passes.
- Firestore clients may only be created for a "demo-*" project with the emulator
  configured. The emulator needs no credentials, so emulator tests keep working.
- The CHEF_ settings from a developer's .env are overridden (environment variables
  win over .env), so providers fail closed if a test forgets a dependency override.

tests/test_hermetic.py proves each guard.
"""

import os
from typing import Any, NoReturn

import google.auth
import google.auth._default
from google.cloud import firestore


class HermeticTestError(RuntimeError):
    """A test tried to reach a real cloud service."""


def _no_credentials(*_args: object, **_kwargs: object) -> NoReturn:
    raise HermeticTestError(
        "tests must not use Google credentials; override the dependency with a fake "
        "or use the Firestore emulator with a demo-* project"
    )


google.auth.default = _no_credentials
google.auth._default.default = _no_credentials  # pyright: ignore[reportPrivateUsage]

_real_firestore_init = firestore.Client.__init__


def _emulator_only_firestore(self: firestore.Client, *args: Any, **kwargs: Any) -> None:
    project = kwargs.get("project", args[0] if args else None)
    if not (isinstance(project, str) and project.startswith("demo-")):
        raise HermeticTestError(f"Firestore client for non-demo project {project!r} in tests")
    if "FIRESTORE_EMULATOR_HOST" not in os.environ:
        raise HermeticTestError("Firestore client in tests without FIRESTORE_EMULATOR_HOST")
    _real_firestore_init(self, *args, **kwargs)


firestore.Client.__init__ = _emulator_only_firestore  # type: ignore[method-assign]

os.environ["CHEF_ENVIRONMENT"] = "test"
os.environ["CHEF_GCP_PROJECT_ID"] = ""
