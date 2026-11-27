# import os
# import tempfile
# import asyncio
# import json
# from contextlib import asynccontextmanager
# from math import gcd
# from pathlib import Path
# from typing import Any, AsyncGenerator

# import numpy as np
# import onnxruntime as ort
# import soundfile as sf
# from scipy.signal import resample_poly
# from fastapi import FastAPI, File, HTTPException, UploadFile
# from fastapi.responses import HTMLResponse, StreamingResponse
# from huggingface_hub import hf_hub_download
# from transformers import WhisperProcessor
# from transformers.models.whisper.tokenization_whisper import LANGUAGES





# # Model files are downloaded from Hugging Face. Audio is only accepted from
# # the upload endpoint and is removed from the temporary directory afterwards.
# # BASE_DIR = Path(__file__).resolve().parent
# # TEMPLATES_DIR = BASE_DIR / "templates"
# # MODEL_DIR = BASE_DIR / "whisper_onnx"
# # REPO_ID = "rav39439/summerization-model"
# # TASK = "transcribe"
# # WHISPER_PROCESSOR_ID = "openai/whisper-base"
# # MAX_GENERATION_TOKENS = 200
# # MAX_UPLOAD_SIZE = 100 * 1024 * 1024
# # SUPPORTED_AUDIO_TYPES = {".wav", ".flac", ".ogg"}
# # SAMPLE_RATE = 16_000
# # N_SAMPLES = SAMPLE_RATE * 30
# # MODEL_DIR = Path("/tmp/models")
# # MODEL_DIR.mkdir(parents=True, exist_ok=True)

# BASE_DIR = Path(__file__).resolve().parent

# TEMPLATES_DIR = BASE_DIR / "templates"

# # Render temporary model directory
# MODEL_DIR = Path("/tmp/models")
# MODEL_DIR.mkdir(parents=True, exist_ok=True)

# REPO_ID = "rav39439/summerization-model"

# TASK = "transcribe"

# WHISPER_PROCESSOR_ID = "openai/whisper-base"

# MAX_GENERATION_TOKENS = 200

# MAX_UPLOAD_SIZE = 100 * 1024 * 1024

# SUPPORTED_AUDIO_TYPES = {".wav", ".flac", ".ogg"}

# SAMPLE_RATE = 16_000

# N_SAMPLES = SAMPLE_RATE * 30

# # INT8 model filenames
# ENCODER_MODEL = "whisper_encoder.onnx"
# DECODER_MODEL = "whisper_decoder.onnx"
# encoder_session: Any = None
# decoder_session: Any = None
# whisper_processor: Any = None


# @asynccontextmanager
# async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
#     load_onnx_models()
#     yield


# app = FastAPI(title="Audio Transcription", lifespan=lifespan)


# def load_onnx_models() -> None:
#     """Download and initialize the Hugging Face ONNX encoder and decoder once."""
#     global encoder_session, decoder_session, whisper_processor
#     if (
#         encoder_session is not None
#         and decoder_session is not None
#         and whisper_processor is not None
#     ):
#         return

    
#     files = [
#         "whisper_encoder.onnx",
#         "whisper_encoder.onnx.data",
#         "whisper_decoder.onnx",
#     ]
#     token = os.getenv("HF_TOKEN") or None
#     for filename in files:
#         hf_hub_download(
#             repo_id=REPO_ID,
#             filename=filename,
#             token=token,
#             local_dir=str(MODEL_DIR),
#         )

#     providers = ["CPUExecutionProvider"]
#     processor = WhisperProcessor.from_pretrained(WHISPER_PROCESSOR_ID)
#     encoder = ort.InferenceSession(
#         str(MODEL_DIR / "whisper_encoder_int8.onnx"), providers=providers
#     )
#     decoder = ort.InferenceSession(
#         str(MODEL_DIR / "whisper_decoder_int8.onnx"), providers=providers
#     )
#     encoder_session = encoder
#     decoder_session = decoder
#     whisper_processor = processor
#     print("Hugging Face ONNX encoder and decoder loaded successfully.")


# @app.get("/", response_class=HTMLResponse)
# async def audio_page() -> HTMLResponse:
#     template_path = TEMPLATES_DIR / "audio.html"
#     if not template_path.is_file():
#         raise HTTPException(status_code=500, detail=f"Template not found: {template_path}")
#     return HTMLResponse(template_path.read_text(encoding="utf-8"))


# def run_encoder(mel: np.ndarray) -> np.ndarray:
#     load_onnx_models()
#     mel = np.asarray(mel, dtype=np.float32)
#     input_name = encoder_session.get_inputs()[0].name
#     return encoder_session.run(None, {input_name: mel})[0]


# def run_decoder(tokens: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
#     load_onnx_models()
#     tokens = np.asarray(tokens, dtype=np.int64)
#     decoder_inputs = decoder_session.get_inputs()
#     input_feed = {
#         decoder_inputs[0].name: tokens,
#         decoder_inputs[1].name: audio_features,
#     }
#     return decoder_session.run(None, input_feed)[0]


# def detect_language(audio_features: np.ndarray) -> str:
#     tokenizer = whisper_processor.tokenizer
#     start_token = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
#     tokens = np.array([[start_token]], dtype=np.int64)
#     logits = run_decoder(tokens, audio_features)
#     language_codes = tuple(LANGUAGES)
#     language_ids = [
#         tokenizer.convert_tokens_to_ids(f"<|{code}|>") for code in language_codes
#     ]
#     language_logits = logits[0, -1, language_ids]
#     best_language_index = int(np.argmax(language_logits))
#     return language_codes[best_language_index]


# def generate_text(audio_features: np.ndarray, language: str) -> str:
#     tokenizer = whisper_processor.tokenizer
#     prefix_tokens = [
#         tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
#         tokenizer.convert_tokens_to_ids(f"<|{language}|>"),
#         tokenizer.convert_tokens_to_ids(f"<|{TASK}|>"),
#     ]
#     end_of_text = tokenizer.eos_token_id
#     generated_tokens = prefix_tokens + [
#         tokenizer.convert_tokens_to_ids("<|notimestamps|>")
#     ]

#     for _ in range(MAX_GENERATION_TOKENS):
#         decoder_input = np.array([generated_tokens], dtype=np.int64)
#         logits = run_decoder(decoder_input, audio_features)
#         next_token_logits = logits[0, -1].copy()
#         next_token_logits[end_of_text + 1:] = -np.inf
#         next_token = int(np.argmax(next_token_logits))
#         if next_token == end_of_text:
#             break
#         generated_tokens.append(next_token)
#         if len(generated_tokens) >= 448:
#             break

#     text_tokens = [
#         token
#         for token in generated_tokens[len(prefix_tokens) + 1:]
#         if token < end_of_text
#     ]
#     return tokenizer.decode(text_tokens).strip()


# def load_audio(audio_path: str) -> np.ndarray:
#     """Read supported audio formats and resample mono audio to Whisper's input rate."""
#     audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
#     if audio.size == 0:
#         raise ValueError("Audio file is empty.")

#     audio = audio.mean(axis=1, dtype=np.float32)
#     if sample_rate != SAMPLE_RATE:
#         divisor = gcd(sample_rate, SAMPLE_RATE)
#         audio = resample_poly(
#             audio, SAMPLE_RATE // divisor, sample_rate // divisor
#         ).astype(np.float32)
#     return audio


# def transcribe_audio(audio_path: str, progress_callback: Any = None) -> tuple[str, str]:
#     """Extract Whisper features and run the existing ONNX prediction flow."""
#     audio = load_audio(audio_path)
#     if len(audio) == 0:
#         raise ValueError("Audio file is empty.")

#     audio_chunks = []
#     detected_language = None
#     chunk_starts = range(0, len(audio), N_SAMPLES)
#     total_chunks = len(chunk_starts)
#     for chunk_index, start in enumerate(chunk_starts, start=1):
#         audio_chunk = audio[start:start + N_SAMPLES]
#         mel = whisper_processor.feature_extractor(
#             audio_chunk,
#             sampling_rate=SAMPLE_RATE,
#             padding="max_length",
#             max_length=N_SAMPLES,
#             truncation=True,
#             return_tensors="np",
#         ).input_features.astype(np.float32)

#         audio_features = run_encoder(mel)
#         if detected_language is None:
#             detected_language = detect_language(audio_features)
#         chunk_text = generate_text(audio_features, detected_language)
#         if chunk_text:
#             audio_chunks.append(chunk_text)
#         if progress_callback:
#             progress_callback(chunk_index, total_chunks)

#     return detected_language or "unknown", " ".join(audio_chunks).strip()


# @app.post("/transcribe")
# async def transcribe(file: UploadFile = File(...)) -> StreamingResponse:
#     suffix = Path(file.filename or "").suffix.lower()
#     if suffix not in SUPPORTED_AUDIO_TYPES:
#         raise HTTPException(
#             status_code=415,
#             detail="Unsupported audio format. Upload a WAV, FLAC, or OGG file.",
#         )

#     temp_path = None
#     try:
#         # Audio decoding needs a path; this temporary upload is always deleted.
#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
#             temp_path = temp_file.name
#             size = 0
#             while chunk := await file.read(1024 * 1024):
#                 size += len(chunk)
#                 if size > MAX_UPLOAD_SIZE:
#                     raise HTTPException(status_code=413, detail="Audio file must be 100 MB or smaller.")
#                 temp_file.write(chunk)

#         if size == 0:
#             raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")

#         audio_path = temp_path
#         temp_path = None

#         async def progress_stream():
#             loop = asyncio.get_running_loop()
#             updates = asyncio.Queue()

#             def report_progress(current: int, total: int) -> None:
#                 loop.call_soon_threadsafe(updates.put_nowait, {"progress": round(current * 100 / total)})

#             async def run_transcription():
#                 try:
#                     language, transcription = await asyncio.to_thread(
#                         transcribe_audio, audio_path, report_progress
#                     )
#                     await updates.put({"progress": 100, "language": language, "transcription": transcription, "done": True})
#                 except Exception as exc:
#                     await updates.put({"error": f"Could not transcribe this audio: {exc}", "done": True})

#             task = asyncio.create_task(run_transcription())
#             try:
#                 while True:
#                     update = await updates.get()
#                     yield json.dumps(update) + "\n"
#                     if update.get("done"):
#                         break
#             finally:
#                 await task
#                 if audio_path and os.path.exists(audio_path):
#                     os.remove(audio_path)

#         return StreamingResponse(progress_stream(), media_type="application/x-ndjson")
#     except HTTPException:
#         raise
#     except Exception as exc:
#         raise HTTPException(status_code=422, detail=f"Could not transcribe this audio: {exc}") from exc
#     finally:
#         await file.close()
#         if temp_path and os.path.exists(temp_path):
#             os.remove(temp_path)




import os
import json
import time
import asyncio
import tempfile
import logging

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from google import genai
from google.genai import types
from dotenv import load_dotenv

# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

TEMPLATES_DIR = BASE_DIR / "templates"

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

MAX_UPLOAD_SIZE = int(
    os.getenv(
        "MAX_UPLOAD_SIZE",
        100 * 1024 * 1024
    )
)

MAX_RETRIES = int(
    os.getenv("GEMINI_MAX_RETRIES", "3")
)

SUPPORTED_AUDIO_TYPES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".aiff"
}

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

gemini_client: Any = None


# --------------------------------------------------
# APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:

    global gemini_client

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        gemini_client = genai.Client(
            api_key=api_key
        )

        logger.info(
            "Gemini API client initialized successfully."
        )

    else:
        logger.warning(
            "GEMINI_API_KEY is missing."
        )

    yield

    if gemini_client is not None:
        gemini_client.close()


app = FastAPI(
    title="Gemini Audio Transcription API",
    lifespan=lifespan
)


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

def get_gemini_client():

    global gemini_client

    if gemini_client is None:

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured in .env."
            )

        gemini_client = genai.Client(
            api_key=api_key
        )

    return gemini_client


# --------------------------------------------------
# GEMINI RETRY FUNCTION
# --------------------------------------------------

def generate_with_retry(
    client,
    model,
    contents,
    config=None,
    max_retries=3
):

    for attempt in range(max_retries + 1):

        try:

            logger.info(
                "Sending request to Gemini model: %s",
                model
            )

            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

        except Exception as exc:

            error_code = getattr(
                exc,
                "code",
                None
            )

            if error_code is None:
                error_code = getattr(
                    exc,
                    "status_code",
                    None
                )

            error_message = str(exc).upper()

            temporary_errors = (
                "503 UNAVAILABLE",
                "429 RESOURCE_EXHAUSTED",
                "500 INTERNAL",
                "502 BAD GATEWAY",
                "504 DEADLINE_EXCEEDED"
            )

            is_temporary = (
                error_code in (429, 500, 502, 503, 504)
                or any(
                    error in error_message
                    for error in temporary_errors
                )
            )

            if not is_temporary:
                raise

            if attempt >= max_retries:
                logger.error(
                    "Gemini retries exhausted: %s",
                    exc
                )
                raise

            wait_time = 2 ** (attempt + 1)

            logger.warning(
                "Gemini temporary error. "
                "Retry %s/%s in %s seconds. Error: %s",
                attempt + 1,
                max_retries,
                wait_time,
                exc
            )

            time.sleep(wait_time)


# --------------------------------------------------
# HTML PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def audio_page():

    template_path = (
        TEMPLATES_DIR / "audio.html"
    )

    if not template_path.is_file():

        raise HTTPException(
            status_code=500,
            detail="Audio HTML template not found."
        )

    return HTMLResponse(
        template_path.read_text(
            encoding="utf-8"
        )
    )


# --------------------------------------------------
# AUDIO TRANSCRIPTION
# --------------------------------------------------

def transcribe_audio(
    audio_path: str,
    progress_callback=None
) -> tuple[str, str]:

    client = get_gemini_client()

    uploaded_file = None

    try:

        # ------------------------------------------
        # STEP 1: UPLOAD AUDIO
        # ------------------------------------------

        if progress_callback:
            progress_callback(15)

        uploaded_file = client.files.upload(
            file=audio_path
        )

        logger.info(
            "Audio uploaded: %s",
            uploaded_file.name
        )

        if progress_callback:
            progress_callback(40)

        # ------------------------------------------
        # STEP 2: TRANSCRIPTION PROMPT
        # ------------------------------------------

        prompt = """
        Transcribe the complete audio recording.

        Instructions:

        1. Automatically identify the spoken language.
        2. Transcribe the speech in its original language.
        3. Do not translate the speech.
        4. Preserve the original meaning and wording.
        5. Include all clearly audible spoken words.
        6. Do not summarize or explain the audio.
        7. Do not invent missing words.
        8. If multiple speakers are distinguishable,
           identify them as Speaker 1, Speaker 2, etc.
        9. Return strictly valid JSON.

        Required JSON format:

        {
            "language": "Detected language",
            "transcription": "Complete transcribed text"
        }
        """

        if progress_callback:
            progress_callback(55)

        # ------------------------------------------
        # STEP 3: GEMINI GENERATION WITH RETRIES
        # ------------------------------------------

        response = generate_with_retry(

            client=client,

            model=MODEL_NAME,

            contents=[
                prompt,
                uploaded_file
            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json"

            ),

            max_retries=MAX_RETRIES
        )

        if progress_callback:
            progress_callback(90)

        # ------------------------------------------
        # STEP 4: PARSE RESPONSE
        # ------------------------------------------

        response_text = response.text

        if not response_text:

            raise RuntimeError(
                "Gemini returned an empty response."
            )

        result = json.loads(response_text)

        language = str(
            result.get(
                "language",
                "unknown"
            )
        )

        transcription = str(
            result.get(
                "transcription",
                ""
            )
        ).strip()

        if not transcription:

            raise RuntimeError(
                "No transcription was returned."
            )

        if progress_callback:
            progress_callback(100)

        return language, transcription

    finally:

        # ------------------------------------------
        # STEP 5: DELETE UPLOADED GEMINI FILE
        # ------------------------------------------

        if uploaded_file is not None:

            try:

                client.files.delete(
                    name=uploaded_file.name
                )

                logger.info(
                    "Gemini temporary audio file deleted."
                )

            except Exception as delete_error:

                logger.warning(
                    "Gemini file deletion warning: %s",
                    delete_error
                )


# --------------------------------------------------
# TRANSCRIPTION ENDPOINT
# --------------------------------------------------

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):

    suffix = Path(
        file.filename or ""
    ).suffix.lower()

    if suffix not in SUPPORTED_AUDIO_TYPES:

        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio format. "
                "Upload WAV, MP3, M4A, AAC, OGG, FLAC or AIFF."
            )
        )

    temp_path = None

    try:

        # ------------------------------------------
        # STEP 1: SAVE TEMPORARY AUDIO
        # ------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            size = 0

            while True:

                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Audio file exceeds the "
                            "maximum allowed upload size."
                        )
                    )

                temp_file.write(chunk)

        if size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty."
            )

        audio_path = temp_path
        temp_path = None

        # ------------------------------------------
        # STEP 2: STREAM PROGRESS
        # ------------------------------------------

        async def progress_stream():

            loop = asyncio.get_running_loop()

            updates = asyncio.Queue()

            def report_progress(value: int):

                loop.call_soon_threadsafe(
                    updates.put_nowait,
                    {"progress": value}
                )

            async def run_transcription():

                try:

                    language, transcription = (
                        await asyncio.to_thread(
                            transcribe_audio,
                            audio_path,
                            report_progress
                        )
                    )

                    await updates.put({

                        "progress": 100,

                        "language": language,

                        "transcription": transcription,

                        "done": True

                    })

                except Exception as exc:

                    logger.exception(
                        "Transcription failed."
                    )

                    await updates.put({

                        "error": (
                            "Could not transcribe this audio: "
                            + str(exc)
                        ),

                        "done": True

                    })

            task = asyncio.create_task(
                run_transcription()
            )

            try:

                while True:

                    update = await updates.get()

                    yield (
                        json.dumps(
                            update,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    if update.get("done"):
                        break

            finally:

                await task

                if os.path.exists(audio_path):
                    os.remove(audio_path)

        return StreamingResponse(
            progress_stream(),
            media_type="application/x-ndjson"
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Audio processing failed."
        )

        raise HTTPException(
            status_code=422,
            detail=f"Could not process audio: {exc}"
        ) from exc

    finally:

        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

