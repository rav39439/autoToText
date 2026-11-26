
# =========================================================
# WHISPER BASE - LOCAL MODELS FOLDER TO ONNX
# Export encoder and decoder separately
# =========================================================

import os
import torch
import whisper
import onnx

from torch import nn
# Disable scaled dot-product attention for ONNX export
import whisper.model as whisper_model

whisper_model.MultiHeadAttention.use_sdpa = False

print("Whisper SDPA disabled for ONNX compatibility.")
# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_NAME = "base"

DEVICE = "cpu"

MODELS_DIR = "./models"

# Complete saved Whisper Base checkpoint
CHECKPOINT_PATH = os.path.join(
    MODELS_DIR,
    "whisper_base_last2.pth"
)

# ONNX output directory
OUTPUT_DIR = os.path.join(
    MODELS_DIR,
    "onnx"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 2. LOAD ORIGINAL WHISPER BASE ARCHITECTURE
# =========================================================

model = whisper.load_model(
    MODEL_NAME,
    device=DEVICE
)

print("Original Whisper Base architecture loaded.")

# =========================================================
# 3. LOAD LOCAL CHECKPOINT
# =========================================================

if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

# =========================================================
# 4. EXTRACT STATE DICTIONARY
# =========================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint

# =========================================================
# 5. LOAD COMPLETE MODEL WEIGHTS
# =========================================================

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print("Complete Whisper Base checkpoint loaded.")

# =========================================================
# 6. ENCODER WRAPPER
# =========================================================

class WhisperEncoderONNX(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel)


# =========================================================
# 7. DECODER WRAPPER
# =========================================================

class WhisperDecoderONNX(nn.Module):

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, tokens, audio_features):

        return self.decoder(
            tokens,
            audio_features
        )


encoder_model = WhisperEncoderONNX(
    model.encoder
).eval()

decoder_model = WhisperDecoderONNX(
    model.decoder
).eval()

# =========================================================
# 8. CREATE DUMMY INPUTS
# =========================================================

n_mels = model.dims.n_mels

audio_context = model.dims.n_audio_ctx

audio_state = model.dims.n_audio_state

dummy_mel = torch.randn(
    1,
    n_mels,
    audio_context * 2,
    dtype=torch.float32
)

dummy_tokens = torch.tensor(
    [[50258, 50259, 50359, 50363]],
    dtype=torch.long
)

dummy_audio_features = torch.randn(
    1,
    audio_context,
    audio_state,
    dtype=torch.float32
)

# =========================================================
# 9. EXPORT ENCODER
# =========================================================

ENCODER_PATH = os.path.join(
    OUTPUT_DIR,
    "whisper_encoder.onnx"
)

print("\nExporting encoder...")

with torch.no_grad():

    torch.onnx.export(
        encoder_model,
        (dummy_mel,),
        ENCODER_PATH,

        input_names=["mel"],
        output_names=["audio_features"],

        dynamic_axes={
            "mel": {
                0: "batch_size"
            },
            "audio_features": {
                0: "batch_size"
            }
        },

        opset_version=17,
        do_constant_folding=True
    )

print("Encoder exported successfully.")

# =========================================================
# 10. EXPORT DECODER
# =========================================================

# =========================================================
# EXPORT WHISPER DECODER TO ONNX
# =========================================================

DECODER_PATH = os.path.join(
    OUTPUT_DIR,
    "whisper_decoder.onnx"
)

print("\nExporting Whisper decoder...")

# Disable the newer dynamo exporter.
# Use the legacy TorchScript-based ONNX exporter.

with torch.no_grad():

    torch.onnx.export(
        decoder_model,

        (
            dummy_tokens,
            dummy_audio_features
        ),

        DECODER_PATH,

        input_names=[
            "tokens",
            "audio_features"
        ],

        output_names=[
            "logits"
        ],

        dynamic_axes={
            "tokens": {
                0: "batch_size",
                1: "sequence_length"
            },

            "audio_features": {
                0: "batch_size"
            },

            "logits": {
                0: "batch_size",
                1: "sequence_length"
            }
        },

        opset_version=17,

        do_constant_folding=True,

        dynamo=False
    )

print("Decoder ONNX exported:", DECODER_PATH)

# =========================================================
# VALIDATE DECODER
# =========================================================

decoder_onnx = onnx.load(DECODER_PATH)

onnx.checker.check_model(decoder_onnx)

print("Decoder ONNX validation successful.")

print(
    "Decoder size:",
    round(
        os.path.getsize(DECODER_PATH) / (1024 * 1024),
        2
    ),
    "MB"
)

# =========================================================
# 12. COMPLETION
# =========================================================

print("\n====================================")
print("WHISPER BASE ONNX CONVERSION DONE")
print("====================================")

print("Encoder:", ENCODER_PATH)
print("Decoder:", DECODER_PATH)
