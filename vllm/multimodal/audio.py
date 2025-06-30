# SPDX-License-Identifier: Apache-2.0
import base64
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt

from vllm.utils import PlaceholderModule

from .base import MediaIO

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

try:
    import soundfile
except ImportError:
    soundfile = PlaceholderModule("soundfile")  # type: ignore[assignment]

try:
    import torch
    import torchaudio
except ImportError:
    torch = PlaceholderModule("torch")  # type: ignore[assignment]


def resample_audio_librosa(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def resample_audio_scipy(
    audio: npt.NDArray[np.floating],
    *,
    orig_sr: float,
    target_sr: float,
):
    # lazy import scipy.signal, otherwise it will crash doc build.
    import scipy.signal

    if orig_sr > target_sr:
        return scipy.signal.resample_poly(audio, 1, orig_sr // target_sr)
    elif orig_sr < target_sr:
        return scipy.signal.resample_poly(audio, target_sr // orig_sr, 1)
    return audio


def resample_audio_torch(
    audio: torch.Tensor,
    *,
    orig_sr: float,
    target_sr: float,
) -> npt.NDArray[np.floating]:
    if not isinstance(audio,torch.Tensor):
        raise RuntimeError("resample_audio_torch need `audio` is type of torch.Tensor")

    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr
    ).to(device=audio.device)

    resampled_audio = resampler(audio).squeeze(0).cpu().numpy()
    return resampled_audio


class AudioResampler:
    """Resample audio data to a target sample rate."""

    def __init__(
        self,
        target_sr: Optional[float] = None,
        method: Literal["librosa", "scipy","torch"] = "librosa",
    ):
        self.target_sr = target_sr
        self.method = method

    def resample(
        self,
        audio: Union[npt.NDArray[np.floating], torch.Tensor],
        *,
        orig_sr: float,
    ) -> npt.NDArray[np.floating]:
        if self.target_sr is None:
            raise RuntimeError("Audio resampling is not supported when "
                               "`target_sr` is not provided")
        if self.method == "librosa":
            return resample_audio_librosa(audio,
                                          orig_sr=orig_sr,
                                          target_sr=self.target_sr)
        elif self.method == "scipy":
            return resample_audio_scipy(audio,
                                        orig_sr=orig_sr,
                                        target_sr=self.target_sr)
        elif self.method == "torch":
            return resample_audio_torch(audio,
                                        orig_sr=orig_sr,
                                        target_sr=self.target_sr)
        else:
            raise ValueError(f"Invalid resampling method: {self.method}. "
                             "Supported methods are 'librosa' and 'scipy'.")


class AudioMediaIO(MediaIO[tuple[npt.NDArray, float]]):

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, float]:
        return librosa.load(BytesIO(data), sr=None)

    def load_base64(
        self,
        media_type: str,
        data: str,
    ) -> tuple[npt.NDArray, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, float]:
        return librosa.load(filepath, sr=None)

    def encode_base64(self, media: tuple[npt.NDArray, float]) -> str:
        audio, sr = media

        with BytesIO() as buffer:
            soundfile.write(buffer, audio, sr, format="WAV")
            data = buffer.getvalue()

        return base64.b64encode(data).decode('utf-8')



class AudioMediaIOWithTorch(MediaIO[torch.Tensor]):

    def load_bytes(self, data: bytes) -> tuple[torch.Tensor, float]:
        # Create a BytesIO object from the input bytes
        byte_stream = BytesIO(data)

        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(byte_stream)

        # Ensure the waveform is in float32 format and move it to CUDA
        audio_tensor = waveform.to(torch.float32).to('cuda')

        return audio_tensor, sample_rate

    def load_base64(
            self,
            media_type: str,
            data: str,
    ) -> tuple[torch.Tensor, float]:
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[torch.Tensor, float]:
        # Load audio using torchaudio
        audio_tensor, sr = torchaudio.load(filepath)
        # Move to GPU using PyTorch
        audio_tensor = audio_tensor.to(torch.float32).to('cuda')
        return audio_tensor, sr

    def encode_base64(self, media: tuple[torch.Tensor, float]) -> str:
        audio_tensor, sr = media
        # Ensure audio is on GPU
        audio_tensor = audio_tensor.to('cuda')

        with BytesIO() as buffer:
            # Process audio on GPU, convert to NumPy for soundfile
            audio_processed = audio_tensor.cpu().numpy()
            soundfile.write(buffer, audio_processed, sr, format="WAV")
            data = buffer.getvalue()
        return base64.b64encode(data).decode('utf-8')