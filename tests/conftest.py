"""Keep the test suite hermetic.

A developer's .env points at the real GCP project. Environment variables win over
.env in pydantic-settings, so blanking the project here makes every provider that
would reach Firestore (or verify real tokens) fail closed with a RuntimeError
instead. A test that forgets a dependency override then fails loudly rather than
writing to production. Emulator tests pass their own demo project explicitly.
"""

import os

os.environ["CHEF_ENVIRONMENT"] = "test"
os.environ["CHEF_GCP_PROJECT_ID"] = ""
