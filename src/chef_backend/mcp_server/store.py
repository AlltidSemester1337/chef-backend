# The Firestore client is only partially typed; results are annotated where they land.
# pyright: reportUnknownMemberType=false
"""Storage for the MCP authorization server: clients, pending logins, codes, tokens.

Codes and tokens are bearer secrets, so only their SHA-256 hash is stored: a leaked
database gives no usable token. They are random 256-bit values, so a plain hash is
enough (no salt or slow hash needed). Every document has an `expireAt` field for a
Firestore TTL policy; reads also check expiry, because TTL deletion is lazy.
"""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

from google.cloud import firestore
from mcp.shared.auth import OAuthClientInformationFull
from pydantic import AnyUrl, BaseModel

CLIENTS_COLLECTION = "mcp_clients"
PENDING_COLLECTION = "mcp_pending"
CODES_COLLECTION = "mcp_codes"
TOKENS_COLLECTION = "mcp_tokens"

TokenKind = Literal["access", "refresh"]


def secret_hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


class PendingAuthorization(BaseModel):
    """An /authorize request waiting for the user to sign in on the login page."""

    client_id: str
    redirect_uri: AnyUrl
    redirect_uri_provided_explicitly: bool
    state: str | None
    scopes: list[str]
    code_challenge: str
    resource: str
    expires_at: float


class StoredCode(BaseModel):
    client_id: str
    uid: str
    scopes: list[str]
    code_challenge: str
    redirect_uri: AnyUrl
    redirect_uri_provided_explicitly: bool
    resource: str
    expires_at: float


class StoredToken(BaseModel):
    kind: TokenKind
    client_id: str
    uid: str
    scopes: list[str]
    resource: str
    # All tokens from one sign-in share a grant: revoking or detecting reuse ends them all.
    grant_id: str
    expires_at: int
    # A refresh token that was exchanged is kept, marked rotated, until it expires:
    # presenting it again means it was stolen (or replayed), so the grant is revoked.
    rotated: bool = False


class OAuthStore(Protocol):
    def save_client(self, client: OAuthClientInformationFull, expires_at: float) -> None: ...
    def get_client(self, client_id: str) -> OAuthClientInformationFull | None: ...

    def save_pending(self, request_id: str, pending: PendingAuthorization) -> None: ...
    def get_pending(self, request_id: str) -> PendingAuthorization | None: ...
    def take_pending(self, request_id: str) -> PendingAuthorization | None:
        """Read and delete in one step, so a pending login completes at most once."""
        ...

    def save_code(self, code: str, stored: StoredCode) -> None: ...
    def get_code(self, code: str) -> StoredCode | None: ...
    def take_code(self, code: str) -> StoredCode | None:
        """Read and delete in one step, so a code is exchanged at most once."""
        ...

    def save_token(self, token: str, stored: StoredToken) -> None: ...
    def get_token(self, token: str) -> StoredToken | None: ...
    def mark_rotated(self, token: str) -> bool:
        """Mark an unrotated refresh token as rotated; False if it already was (or is gone)."""
        ...

    def delete_grant(self, grant_id: str) -> None: ...


def _expire_at(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, UTC)


class FirestoreOAuthStore:
    def __init__(self, client: firestore.Client) -> None:
        self._client = client

    def _doc(self, collection: str, key: str) -> Any:
        return self._client.collection(collection).document(key)

    def _read(self, collection: str, key: str) -> dict[str, Any] | None:
        snapshot = self._doc(collection, key).get()
        return cast(dict[str, Any] | None, snapshot.to_dict())

    def _take(self, collection: str, key: str) -> dict[str, Any] | None:
        ref = self._doc(collection, key)

        @firestore.transactional
        def take(transaction: firestore.Transaction) -> dict[str, Any] | None:
            data = cast(dict[str, Any] | None, ref.get(transaction=transaction).to_dict())
            if data is not None:
                transaction.delete(ref)
            return data

        return cast(Callable[[firestore.Transaction], dict[str, Any] | None], take)(
            self._client.transaction()
        )

    # --- clients -------------------------------------------------------------

    def save_client(self, client: OAuthClientInformationFull, expires_at: float) -> None:
        if client.client_id is None:
            raise ValueError("cannot store a client without an id")
        data = client.model_dump(mode="json", exclude_none=True)
        self._doc(CLIENTS_COLLECTION, client.client_id).set(
            {"client": data, "expireAt": _expire_at(expires_at)}
        )

    def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        data = self._read(CLIENTS_COLLECTION, client_id)
        if data is None or data["expireAt"] <= datetime.now(UTC):
            return None
        return OAuthClientInformationFull.model_validate(data["client"])

    # --- pending logins ------------------------------------------------------

    def save_pending(self, request_id: str, pending: PendingAuthorization) -> None:
        self._doc(PENDING_COLLECTION, secret_hash(request_id)).set(
            pending.model_dump(mode="json") | {"expireAt": _expire_at(pending.expires_at)}
        )

    def get_pending(self, request_id: str) -> PendingAuthorization | None:
        data = self._read(PENDING_COLLECTION, secret_hash(request_id))
        return PendingAuthorization.model_validate(data) if data is not None else None

    def take_pending(self, request_id: str) -> PendingAuthorization | None:
        data = self._take(PENDING_COLLECTION, secret_hash(request_id))
        return PendingAuthorization.model_validate(data) if data is not None else None

    # --- codes ---------------------------------------------------------------

    def save_code(self, code: str, stored: StoredCode) -> None:
        self._doc(CODES_COLLECTION, secret_hash(code)).set(
            stored.model_dump(mode="json") | {"expireAt": _expire_at(stored.expires_at)}
        )

    def get_code(self, code: str) -> StoredCode | None:
        data = self._read(CODES_COLLECTION, secret_hash(code))
        return StoredCode.model_validate(data) if data is not None else None

    def take_code(self, code: str) -> StoredCode | None:
        data = self._take(CODES_COLLECTION, secret_hash(code))
        return StoredCode.model_validate(data) if data is not None else None

    # --- tokens --------------------------------------------------------------

    def save_token(self, token: str, stored: StoredToken) -> None:
        self._doc(TOKENS_COLLECTION, secret_hash(token)).set(
            stored.model_dump(mode="json") | {"expireAt": _expire_at(stored.expires_at)}
        )

    def get_token(self, token: str) -> StoredToken | None:
        data = self._read(TOKENS_COLLECTION, secret_hash(token))
        return StoredToken.model_validate(data) if data is not None else None

    def mark_rotated(self, token: str) -> bool:
        ref = self._doc(TOKENS_COLLECTION, secret_hash(token))

        @firestore.transactional
        def mark(transaction: firestore.Transaction) -> bool:
            data = cast(dict[str, Any] | None, ref.get(transaction=transaction).to_dict())
            if data is None or data.get("rotated"):
                return False
            transaction.update(ref, {"rotated": True})
            return True

        return cast(Callable[[firestore.Transaction], bool], mark)(self._client.transaction())

    def delete_grant(self, grant_id: str) -> None:
        query = self._client.collection(TOKENS_COLLECTION).where(
            filter=firestore.FieldFilter("grant_id", "==", grant_id)
        )
        batch = self._client.batch()
        for snapshot in query.stream():
            batch.delete(snapshot.reference)
        batch.commit()
