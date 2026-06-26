# Copyright 2023 The HuggingFace Inc. team and the librosa & torchaudio authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Audio processing functions to extract features from audio waveforms. This code is pure numpy to support all frameworks
and remove unnecessary dependencies.
"""

# ... (Header licence inchangé)

import base64
import io
import os
import warnings
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import requests

from utils import (
    is_librosa_available,
    is_numpy_array,
    is_soundfile_available,
    is_torch_tensor,
    requires_backends,
)

if is_soundfile_available():
    import soundfile as sf

if is_librosa_available():
    import librosa
    import soxr

def load_audio(audio: Union[str, np.ndarray], sampling_rate=16000, timeout=None) -> np.ndarray:
    requires_backends(load_audio, ["librosa"])
    if isinstance(audio, str):
        if audio.startswith("http://") or audio.startswith("https://"):
            audio = librosa.load(BytesIO(requests.get(audio, timeout=timeout).content), sr=sampling_rate)[0]
        elif os.path.isfile(audio):
            audio = librosa.load(audio, sr=sampling_rate)[0]
    elif isinstance(audio, np.ndarray):
        audio = audio
    else:
        raise TypeError("Incorrect format used for `audio`. Should be an url linking to an audio, a local path, or numpy array.")
    return audio

def load_audio_as(
    audio: str,
    return_format: str,
    timeout: Optional[int] = None,
    force_mono: bool = False,
    sampling_rate: Optional[int] = None,
) -> Union[str, dict[str, Any], io.BytesIO, None]:
    requires_backends(load_audio_as, ["librosa"])
    if return_format not in ["base64", "dict", "buffer"]:
        raise ValueError(f"Invalid return_format: {return_format}. Must be 'base64', 'dict', or 'buffer'")
    try:
        audio_bytes = None
        if audio.startswith(("http://", "https://")):
            response = requests.get(audio, timeout=timeout)
            response.raise_for_status()
            audio_bytes = response.content
        elif os.path.isfile(audio):
            with open(audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        else:
            raise ValueError(f"File not found: {audio}")

        with io.BytesIO(audio_bytes) as audio_file:
            with sf.SoundFile(audio_file) as f:
                audio_array = f.read(dtype="float32")
                original_sr = f.samplerate
                audio_format = f.format
                if sampling_rate is not None and sampling_rate != original_sr:
                    audio_array = soxr.resample(audio_array, original_sr, sampling_rate, quality="HQ")
                else:
                    sampling_rate = original_sr

        if force_mono and audio_array.ndim != 1:
            audio_array = audio_array.mean(axis=1)
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sampling_rate, format=audio_format.upper())
        buffer.seek(0)
        if return_format == "buffer":
            return buffer
        elif return_format == "base64":
            return base64.b64encode(buffer.read()).decode("utf-8")
        elif return_format == "dict":
            return {
                "data": base64.b64encode(buffer.read()).decode("utf-8"),
                "format": audio_format.lower(),
            }
    except Exception as e:
        raise ValueError(f"Error loading audio: {e}")

AudioInput = Union[
    np.ndarray, "torch.Tensor", list[np.ndarray], tuple[np.ndarray], list["torch.Tensor"], tuple["torch.Tensor"]
]

def is_valid_audio(audio):
    return is_numpy_array(audio) or is_torch_tensor(audio)

def is_valid_list_of_audio(audio):
    return audio and all(is_valid_audio(audio_i) for audio_i in audio)

def make_list_of_audio(
    audio: Union[list[AudioInput], AudioInput],
) -> AudioInput:
    if isinstance(audio, (list, tuple)) and is_valid_list_of_audio(audio):
        return audio
    if is_valid_audio(audio):
        return [audio]
    raise ValueError("Invalid input type. Must be a single audio or a list of audio")

def hertz_to_mel(freq: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    elif mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    mels = 3.0 * freq / 200.0
    if isinstance(freq, np.ndarray):
        log_region = freq >= min_log_hertz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    elif freq >= min_log_hertz:
        mels = min_log_mel + np.log(freq / min_log_hertz) * logstep
    return mels

def mel_to_hertz(mels: Union[float, np.ndarray], mel_scale: str = "htk") -> Union[float, np.ndarray]:
    if mel_scale not in ["slaney", "htk", "kaldi"]:
        raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')
    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    elif mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    freq = 200.0 * mels / 3.0
    if isinstance(mels, np.ndarray):
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    elif mels >= min_log_mel:
        freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))
    return freq

def hertz_to_octave(
    freq: Union[float, np.ndarray], tuning: Optional[float] = 0.0, bins_per_octave: Optional[int] = 12
):
    stuttgart_pitch = 440.0 * 2.0 ** (tuning / bins_per_octave)
    octave = np.log2(freq / (float(stuttgart_pitch) / 16))
    return octave

def _create_triangular_filter_bank(fft_freqs: np.ndarray, filter_freqs: np.ndarray) -> np.ndarray:
    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

def chroma_filter_bank(
    num_frequency_bins: int,
    num_chroma: int,
    sampling_rate: int,
    tuning: float = 0.0,
    power: Optional[float] = 2.0,
    weighting_parameters: Optional[tuple[float, float]] = (5.0, 2.0),
    start_at_c_chroma: Optional[bool] = True,
):
    frequencies = np.linspace(0, sampling_rate, num_frequency_bins, endpoint=False)[1:]
    freq_bins = num_chroma * hertz_to_octave(frequencies, tuning=tuning, bins_per_octave=num_chroma)
    freq_bins = np.concatenate(([freq_bins[0] - 1.5 * num_chroma], freq_bins))
    bins_width = np.concatenate((np.maximum(freq_bins[1:] - freq_bins[:-1], 1.0), [1]))
    chroma_filters = np.subtract.outer(freq_bins, np.arange(0, num_chroma, dtype="d")).T
    num_chroma2 = np.round(float(num_chroma) / 2)
    chroma_filters = np.remainder(chroma_filters + num_chroma2 + 10 * num_chroma, num_chroma) - num_chroma2
    chroma_filters = np.exp(-0.5 * (2 * chroma_filters / np.tile(bins_width, (num_chroma, 1))) ** 2)
    if power is not None:
        chroma_filters = chroma_filters / np.sum(chroma_filters**power, axis=0, keepdims=True) ** (1.0 / power)
    if weighting_parameters is not None:
        center, half_width = weighting_parameters
        chroma_filters *= np.tile(
            np.exp(-0.5 * (((freq_bins / num_chroma - center) / half_width) ** 2)),
            (num_chroma, 1),
        )
    if start_at_c_chroma:
        chroma_filters = np.roll(chroma_filters, -3 * (num_chroma // 12), axis=0)
    return np.ascontiguousarray(chroma_filters[:, : int(1 + num_frequency_bins / 2)])

def mel_filter_bank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
    norm: Optional[str] = None,
    mel_scale: str = "htk",
    triangularize_in_mel_space: bool = False,
) -> np.ndarray:
    if norm is not None and norm != "slaney":
        raise ValueError('norm must be one of None or "slaney"')
    if num_frequency_bins < 2:
        raise ValueError(f"Require num_frequency_bins: {num_frequency_bins} >= 2")
    if min_frequency > max_frequency:
        raise ValueError(f"Require min_frequency: {min_frequency} <= max_frequency: {max_frequency}")
    mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)
    if triangularize_in_mel_space:
        fft_bin_width = sampling_rate / ((num_frequency_bins - 1) * 2)
        fft_freqs = hertz_to_mel(fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale)
        filter_freqs = mel_freqs
    else:
        fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)
    mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)
    if norm is not None and norm == "slaney":
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)
    if (mel_filters.max(axis=0) == 0.0).any():
        warnings.warn(
            "At least one mel filter has all zero values. "
            f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
            f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
        )
    return mel_filters

def optimal_fft_length(window_length: int) -> int:
    return 2 ** int(np.ceil(np.log2(window_length)))

def window_function(
    window_length: int,
    name: str = "hann",
    periodic: bool = True,
    frame_length: Optional[int] = None,
    center: bool = True,
) -> np.ndarray:
    length = window_length + 1 if periodic else window_length
    if name == "boxcar":
        window = np.ones(length)
    elif name in ["hamming", "hamming_window"]:
        window = np.hamming(length)
    elif name in ["hann", "hann_window"]:
        window = np.hanning(length)
    elif name in ["povey"]:
        window = np.power(np.hanning(length), 0.85)
    else:
        raise ValueError(f"Unknown window function '{name}'")
    if periodic:
        window = window[:-1]
    if frame_length is None:
        return window
    if window_length > frame_length:
        raise ValueError(
            f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
        )
    padded_window = np.zeros(frame_length)
    offset = (frame_length - window_length) // 2 if center else 0
    padded_window[offset : offset + window_length] = window
    return padded_window

def spectrogram(
    waveform: np.ndarray,
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    dither: float = 0.0,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    remove_dc_offset: Optional[bool] = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    window_length = len(window)
    if fft_length is None:
        fft_length = frame_length
    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")
    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")
    if waveform.ndim != 1:
        raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")
    if np.iscomplexobj(waveform):
        raise ValueError("Complex-valued input waveforms are not currently supported")
    if power is None and mel_filters is not None:
        raise ValueError(
            "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram."
            "Specify `power` to fix this issue."
        )
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform = np.pad(waveform, padding, mode=pad_mode)
    waveform = waveform.astype(np.float64)
    window = window.astype(np.float64)
    num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))
    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros(fft_length)
    timestep = 0
    for frame_idx in range(num_frames):
        buffer[:frame_length] = waveform[timestep : timestep + frame_length]
        if dither != 0.0:
            buffer[:frame_length] += dither * np.random.randn(frame_length)
        if remove_dc_offset:
            buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()
        if preemphasis is not None:
            buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
            buffer[0] *= 1 - preemphasis
        buffer[:frame_length] *= window
        spectrogram[frame_idx] = fft_func(buffer)
        timestep += hop_length
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power
    spectrogram = spectrogram.T
    if mel_filters is not None:
        spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))
    if power is not None and log_mel is not None:
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        elif log_mel == "dB":
            if power == 1.0:
                spectrogram = amplitude_to_db(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")
        spectrogram = np.asarray(spectrogram, dtype)
    return spectrogram

def spectrogram_batch(
    waveform_list: list[np.ndarray],
    window: np.ndarray,
    frame_length: int,
    hop_length: int,
    fft_length: Optional[int] = None,
    power: Optional[float] = 1.0,
    center: bool = True,
    pad_mode: str = "reflect",
    onesided: bool = True,
    dither: float = 0.0,
    preemphasis: Optional[float] = None,
    mel_filters: Optional[np.ndarray] = None,
    mel_floor: float = 1e-10,
    log_mel: Optional[str] = None,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
    remove_dc_offset: Optional[bool] = None,
    dtype: np.dtype = np.float32,
) -> list[np.ndarray]:
    window_length = len(window)
    if fft_length is None:
        fft_length = frame_length
    if frame_length > fft_length:
        raise ValueError(f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})")
    if window_length != frame_length:
        raise ValueError(f"Length of the window ({window_length}) must equal frame_length ({frame_length})")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")
    for waveform in waveform_list:
        if waveform.ndim != 1:
            raise ValueError(f"Input waveform must have only one dimension, shape is {waveform.shape}")
        if np.iscomplexobj(waveform):
            raise ValueError("Complex-valued input waveforms are not currently supported")
    if center:
        padding = [(int(frame_length // 2), int(frame_length // 2))]
        waveform_list = [
            np.pad(waveform, padding, mode=pad_mode) for waveform in waveform_list
        ]
    original_waveform_lengths = [len(waveform) for waveform in waveform_list]
    max_length = max(original_waveform_lengths)
    padded_waveform_batch = np.array(
        [
            np.pad(waveform, (0, max_length - len(waveform)), mode="constant", constant_values=0)
            for waveform in waveform_list
        ],
        dtype=dtype,
    )
    padded_waveform_batch = padded_waveform_batch.astype(np.float64)
    window = window.astype(np.float64)
    num_frames = int(1 + np.floor((padded_waveform_batch.shape[1] - frame_length) / hop_length))
    true_num_frames = [int(1 + np.floor((length - frame_length) / hop_length)) for length in original_waveform_lengths]
    num_batches = padded_waveform_batch.shape[0]
    num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
    spectrogram = np.empty((num_batches, num_frames, num_frequency_bins), dtype=np.complex64)
    fft_func = np.fft.rfft if onesided else np.fft.fft
    buffer = np.zeros((num_batches, fft_length))
    for frame_idx in range(num_frames):
        timestep = frame_idx * hop_length
        buffer[:, :frame_length] = padded_waveform_batch[:, timestep : timestep + frame_length]
        if dither != 0.0:
            buffer[:, :frame_length] += dither * np.random.randn(*buffer[:, :frame_length].shape)
        if remove_dc_offset:
            buffer[:, :frame_length] -= buffer[:, :frame_length].mean(axis=1, keepdims=True)
        if preemphasis is not None:
            buffer[:, 1:frame_length] -= preemphasis * buffer[:, : frame_length - 1]
            buffer[:, 0] *= 1 - preemphasis
        buffer[:, :frame_length] *= window
        spectrogram[:, frame_idx] = fft_func(buffer)
    if power is not None:
        spectrogram = np.abs(spectrogram, dtype=np.float64) ** power
    if mel_filters is not None:
        result = np.tensordot(spectrogram, mel_filters.T, axes=([2], [1]))
        spectrogram = np.maximum(mel_floor, result)
    if power is not None and log_mel is not None:
        if log_mel == "log":
            spectrogram = np.log(spectrogram)
        elif log_mel == "log10":
            spectrogram = np.log10(spectrogram)
        elif log_mel == "dB":
            if power == 1.0:
                spectrogram = amplitude_to_db_batch(spectrogram, reference, min_value, db_range)
            elif power == 2.0:
                spectrogram = power_to_db_batch(spectrogram, reference, min_value, db_range)
            else:
                raise ValueError(f"Cannot use log_mel option '{log_mel}' with power {power}")
        else:
            raise ValueError(f"Unknown log_mel option: {log_mel}")
        spectrogram = np.asarray(spectrogram, dtype)
    spectrogram_list = [spectrogram[i, : true_num_frames[i], :].T for i in range(len(true_num_frames))]
    return spectrogram_list

def power_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
) -> np.ndarray:
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")
    reference = max(min_value, reference)
    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))
    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)
    return spectrogram

def power_to_db_batch(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-10,
    db_range: Optional[float] = None,
) -> np.ndarray:
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")
    reference = max(min_value, reference)
    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))
    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        max_values = spectrogram.max(axis=(1, 2), keepdims=True)
        spectrogram = np.clip(spectrogram, a_min=max_values - db_range, a_max=None)
    return spectrogram

def amplitude_to_db(
    spectrogram: np.ndarray,
    reference: float = 1.0,
    min_value: float = 1e-5,
    db_range: Optional[float] = None,
) -> np.ndarray:
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")
    reference = max(min_value, reference)
    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))
    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        spectrogram = np.clip(spectrogram, a_min=spectrogram.max() - db_range, a_max=None)
    return spectrogram

def amplitude_to_db_batch(
    spectrogram: np.ndarray, reference: float = 1.0, min_value: float = 1e-5, db_range: Optional[float] = None
) -> np.ndarray:
    if reference <= 0.0:
        raise ValueError("reference must be greater than zero")
    if min_value <= 0.0:
        raise ValueError("min_value must be greater than zero")
    reference = max(min_value, reference)
    spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
    spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))
    if db_range is not None:
        if db_range <= 0.0:
            raise ValueError("db_range must be greater than zero")
        max_values = spectrogram.max(axis=(1, 2), keepdims=True)
        spectrogram = np.clip(spectrogram, a_min=max_values - db_range, a_max=None)
    return spectrogram
def preprocess_audio(
    audio_path,
    sampling_rate=16000,
    mono=True,
    normalize=True,
    timeout=None
):
    """
    Prétraitement audio avancé :
    - Charge l'audio avec sampling rate voulu
    - Convertit en mono si demandé
    - Normalise le volume entre -1 et 1 si demandé
    - Peut charger depuis URL ou chemin local

    Args :
        audio_path : Chemin (str), URL ou ndarray
        sampling_rate : int
        mono : bool
        normalize : bool
        timeout : float (pour récupération URL)

    Returns :
        np.ndarray
    """
    y = load_audio(audio_path, sampling_rate=sampling_rate, timeout=timeout)
    if mono and isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=0) if y.shape[0] < y.shape[1] else y.mean(axis=1)
    if normalize and isinstance(y, np.ndarray):
        max_abs = np.max(np.abs(y))
        if max_abs > 0:
            y = y / max_abs
    return y
