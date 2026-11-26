"""Inference for the original MusicFM-MSD release, using its saved mel buffers and weights."""

import json
import math
import time
from pathlib import Path

import numpy as np

from music_atlas.split_views import digest

CHECKPOINT = "minzwon/MusicFM"
REVISION = "4513b38bc25ad1d227b1980819b9691ba97f4d87"
CONFIG_REPO = "facebook/wav2vec2-conformer-rope-large-960h-ft"
CONFIG_REVISION = "c85b633f0428fddd34b1b8fa3a8559019dd47f3d"
WEIGHTS_SHA256 = "218b483a0256ddef736267425fabb166fd97008983696bb9270def464b47bded"
RATE = 24_000
DIMENSIONS = 1024
LAYER = 7
FFT = 2048
HOP = 240


def checkpoint_files(*, offline):
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    def file(repo, revision, name):
        return Path(hf_hub_download(repo, name, revision=revision, local_files_only=offline))

    try:
        return (
            file(CHECKPOINT, REVISION, "pretrained_msd.pt"),
            file(CHECKPOINT, REVISION, "msd_stats.json"),
            file(CONFIG_REPO, CONFIG_REVISION, "config.json"),
        )
    except LocalEntryNotFoundError as error:
        raise ValueError(
            "MusicFM-MSD files are missing. Run prepare-musicfm on a CPU node."
        ) from error


def prepare():
    started = time.perf_counter()
    weights, _, _ = checkpoint_files(offline=False)
    fetched = time.perf_counter()
    if digest(weights) != WEIGHTS_SHA256:
        raise ValueError("MusicFM-MSD checkpoint SHA-256 differs from the published value.")
    encode, _ = load_encoder("cpu")
    phase = np.arange(10 * RATE, dtype=np.float32) * (2 * np.pi / RATE)
    probe = 0.1 * np.sin(np.array([[220], [440]], np.float32) * phase)
    vectors = encode(probe)
    if vectors.shape != (2, DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError("MusicFM-MSD CPU smoke check returned invalid vectors.")
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12) or np.allclose(vectors[0], vectors[1]):
        raise ValueError("MusicFM-MSD CPU smoke check returned degenerate vectors.")
    print(
        f"Cached {CHECKPOINT}@{REVISION} in {fetched - started:.1f}s. "
        f"Verified {weights.name} SHA-256 and encoded two synthetic CPU probes.",
        flush=True,
    )


def _substate(state, prefix):
    return {
        name[len(prefix) :]: tensor for name, tensor in state.items() if name.startswith(prefix)
    }


def _encoder(state, stats, config, device):
    import torch
    from torch import nn
    from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
        Wav2Vec2ConformerEncoder,
    )

    # Inference-only form of the MIT-licensed MusicFM frontend (ByteDance, 2023).
    # https://github.com/minzwon/musicfm/blob/master/modules/conv.py
    class Residual(nn.Module):
        def __init__(self, n_in, width):
            super().__init__()
            self.conv1 = nn.Conv2d(n_in, width, 3, stride=2, padding=1)
            self.conv2 = nn.Conv2d(width, width, 3, padding=1)
            self.conv3 = nn.Conv2d(n_in, width, 3, stride=2, padding=1)
            self.bn1, self.bn2, self.bn3 = (nn.BatchNorm2d(width) for _ in range(3))

        def forward(self, x):
            main = self.bn2(self.conv2(self.bn1(self.conv1(x)).relu()))
            return (main + self.bn3(self.conv3(x))).relu()

    conv_state = _substate(state, "conv.")
    width = conv_state["conv.0.conv1.weight"].shape[0]
    frontend = nn.ModuleDict(
        {
            "conv": nn.Sequential(Residual(1, width), Residual(width, width)),
            "linear": nn.Linear(width * 32, config.hidden_size),
        }
    )
    frontend.load_state_dict(conv_state, strict=True)
    frontend.to(device).eval()
    conformer = Wav2Vec2ConformerEncoder(config)
    conformer.load_state_dict(_substate(state, "conformer."), strict=True)
    conformer.to(device).eval()

    mel = _substate(state, "preprocessor_melspec_2048.mel_stft.")
    window = mel["spectrogram.window"].to(device=device, dtype=torch.float32)
    filters = mel["mel_scale.fb"].to(device=device, dtype=torch.float32)
    mean, std = stats["melspec_2048_mean"], stats["melspec_2048_std"]
    if (
        window.shape != (FFT,)
        or filters.shape != (FFT // 2 + 1, 128)
        or not math.isfinite(mean)
        or not math.isfinite(std)
        or std <= 0
    ):
        raise ValueError("MusicFM-MSD mel buffers or statistics do not match the published recipe.")

    @torch.inference_mode()
    def encode(samples):
        if samples.ndim != 2 or samples.shape[1] != 10 * RATE:
            raise ValueError("MusicFM-MSD expects shared 10-second mono windows at 24 kHz.")
        waveform = torch.as_tensor(samples, dtype=torch.float32, device=device)
        spectrum = torch.stft(waveform, FFT, HOP, window=window, return_complex=True)
        mel_power = (spectrum.abs().square().transpose(1, 2) @ filters).transpose(1, 2)
        # The reference removes the final centred STFT frame before its residual blocks.
        db = 10 * torch.log10(mel_power[..., :-1].clamp_min(1e-10))
        features = frontend["conv"](((db - mean) / std).unsqueeze(1))
        sequence = frontend["linear"](features.permute(0, 3, 1, 2).flatten(2))
        frames = conformer(sequence, output_hidden_states=True).hidden_states[LAYER]
        return frames.mean(dim=1).float().cpu().numpy()

    return encode


def load_encoder(device):
    import torch
    from transformers import Wav2Vec2ConformerConfig

    start = time.perf_counter()
    weights_path, stats_path, config_path = checkpoint_files(offline=True)
    snapshot = time.perf_counter()
    config = Wav2Vec2ConformerConfig.from_json_file(config_path)
    config.num_hidden_layers = 12
    if config.hidden_size != DIMENSIONS or config.position_embeddings_type != "rotary":
        raise ValueError("Unexpected MusicFM Conformer configuration.")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)["state_dict"]
    if not all(name.startswith("model.") for name in weights):
        raise ValueError("Unexpected MusicFM-MSD checkpoint parameter names.")
    state = {key.removeprefix("model."): value for key, value in weights.items()}
    encode = _encoder(state, stats, config, device)
    return encode, {
        "checkpoint_cache_hit": True,
        "checkpoint_fetch_seconds": 0.0,
        "snapshot_seconds": snapshot - start,
        "model_init_seconds": time.perf_counter() - snapshot,
    }


if __name__ == "__main__":
    prepare()
