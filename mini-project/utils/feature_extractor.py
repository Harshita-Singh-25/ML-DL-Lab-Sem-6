import librosa

import numpy as np

def extract_flat_features(filepath):
    """For SVM — flat feature vector"""
    try:
        y, sr = librosa.load(filepath, duration=3, offset=0.5)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        
        features = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.mean(mel, axis=1),
            [zcr], [rms]
        ])
        return features
    except:
        return None

def extract_flat_features_from_audio(y, sr):
    """For SVM — from raw audio array"""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        
        features = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(chroma, axis=1), np.mean(mel, axis=1),
            [zcr], [rms]
        ])
        return features
    except:
        return None

def extract_lstm_features(y, sr, max_len=130):
    """For LSTM — MFCC sequence"""
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T
        if mfcc.shape[0] < max_len:
            mfcc = np.pad(mfcc, ((0, max_len - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:max_len, :]
        return mfcc
    except:
        return None

def extract_cnn_features(y, sr, max_len=128):
    """For CNN — mel spectrogram image"""
    try:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        if mel_db.shape[1] < max_len:
            mel_db = np.pad(mel_db, ((0,0),(0, max_len - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :max_len]
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_db
    except:
        return None

def get_waveform_data(y, sr):
    """Returns waveform for visualization"""
    times = np.linspace(0, len(y)/sr, len(y))
    return times, y

def get_spectrogram_data(y, sr):
    """Returns spectrogram for visualization"""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db