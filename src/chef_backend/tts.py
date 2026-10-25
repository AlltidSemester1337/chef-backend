"""Text-to-speech proxy: Google Cloud TTS called as the service account (ADC, no key).

Voice and audio format are fixed here, so clients cannot pick a pricier voice.
The apps send one sentence per request; the length cap matches theirs.
"""

import logging
from datetime import date
from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, Response, status
from google.api_core.exceptions import GoogleAPIError
from google.cloud import texttospeech
from pydantic import BaseModel, ConfigDict, Field

from chef_backend.access import require_budget, require_quota
from chef_backend.ai.client import UpstreamError
from chef_backend.ai.dependencies import (
    get_beta_interaction_limit,
    get_budget_store,
    get_daily_tts_char_limit,
    get_quota_store,
    get_today,
)
from chef_backend.auth import AuthenticatedUser, get_current_user
from chef_backend.budget import BudgetStore, UsageKind, check_budget
from chef_backend.quota import QuotaStore, check_quota

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 4500
LANGUAGE_CODE = "en-US"
VOICE_NAME = "en-US-Chirp3-HD-Aoede"
TIMEOUT_SECONDS = 30.0


class SynthesizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[str, Field(min_length=1, max_length=MAX_TTS_CHARS)]


class SpeechSynthesizer(Protocol):
    def synthesize(self, text: str) -> bytes:
        """Return MP3 audio for the text, or raise UpstreamError."""
        ...


class GoogleSpeechSynthesizer:
    def __init__(self, client: texttospeech.TextToSpeechClient) -> None:
        self._client = client

    def synthesize(self, text: str) -> bytes:
        try:
            response = self._client.synthesize_speech(  # pyright: ignore[reportUnknownMemberType]
                input=texttospeech.SynthesisInput(text=text),
                voice=texttospeech.VoiceSelectionParams(
                    language_code=LANGUAGE_CODE, name=VOICE_NAME
                ),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3
                ),
                timeout=TIMEOUT_SECONDS,
            )
        except GoogleAPIError as error:
            raise UpstreamError(f"TTS request failed: {error!r}") from error
        return response.audio_content


@lru_cache
def get_speech_synthesizer() -> SpeechSynthesizer:
    # Credentials via ADC: the Cloud Run service account, locally gcloud's user login.
    return GoogleSpeechSynthesizer(texttospeech.TextToSpeechClient())


router = APIRouter(prefix="/v1/tts", tags=["tts"])


@router.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}, "description": "MP3 audio"}},
)
def synthesize(
    request: SynthesizeRequest,
    user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    synthesizer: Annotated[SpeechSynthesizer, Depends(get_speech_synthesizer)],
    quota_store: Annotated[QuotaStore, Depends(get_quota_store)],
    limit: Annotated[int, Depends(get_beta_interaction_limit)],
    budget_store: Annotated[BudgetStore, Depends(get_budget_store)],
    daily_char_limit: Annotated[int, Depends(get_daily_tts_char_limit)],
    today: Annotated[date, Depends(get_today)],
) -> Response:
    """Speak one piece of text. Needs an eligible user and fits in a daily character
    budget (the apps send one sentence per request, so calls would be a poor measure)."""
    require_quota(check_quota(user, quota_store, counted=False, limit=limit))
    require_budget(
        check_budget(
            user,
            budget_store,
            kind=UsageKind.TTS_CHARS,
            amount=len(request.text),
            limit=daily_char_limit,
            day=today,
        )
    )

    try:
        audio = synthesizer.synthesize(request.text)
    except UpstreamError:
        logger.exception("TTS call failed")
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "upstream_error") from None

    return Response(content=audio, media_type="audio/mpeg")
