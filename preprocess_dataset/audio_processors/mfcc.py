import librosa
import librosa.display
import numpy
import scipy.fftpack

from audio_processors.consts import (
    N_COEFF,
    WINDOW_SIZE,
    SAMPLING_RATE,
)


def extract_cc(audio_series, window_size, filter_basis):
    windowed = librosa.core.stft(
        audio_series, window_size,
        window='hamming', dtype=numpy.float32
    )
    filtered = numpy.dot(filter_basis, windowed)
    energy = librosa.core.power_to_db(filtered)
    cepstral_coeff = scipy.fftpack.dct(energy, norm='ortho')[:N_COEFF]
    return cepstral_coeff


def extract_mfcc(audio_series, sampling_rate):
    filter_basis = librosa.filters.mel(sampling_rate, WINDOW_SIZE)
    mfcc = extract_cc(audio_series, WINDOW_SIZE, filter_basis)
    return mfcc


def process_file(filename):
    audio_series, sampling_rate = librosa.load(filename, duration=25, sr=SAMPLING_RATE)
    mfcc = extract_mfcc(audio_series, sampling_rate)
    return mfcc
