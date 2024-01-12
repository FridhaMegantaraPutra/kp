import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
import pandas as pd
import joblib
import pickle
from scipy.stats import kurtosis
from scipy.stats import skew
import sounddevice as sd
import soundfile as sf
import tempfile

# Load the previously saved StandardScaler object
scaler = joblib.load('scaler.pkl')

# Load model from the pickle file
with open('mlp_model.pkl', 'rb') as model_file:
    saved_model = pickle.load(model_file)


def calculate_statistics(y, sr):
    # menghilangkan noise dan silent part
    y = librosa.effects.preemphasis(y)
    y_trimmed, index = librosa.effects.trim(y, top_db=20)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)

    # UNTUK MENGHITUNG NILAI ZCR
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    zcr_median = np.median(librosa.feature.zero_crossing_rate(y=y))
    zcr_std_dev = np.std(librosa.feature.zero_crossing_rate(y=y))
    zcr_kurtosis = kurtosis(librosa.feature.zero_crossing_rate(y=y)[0])
    zcr_skew = skew(librosa.feature.zero_crossing_rate(y=y)[0])

    # UNTUK MENGHITUNG NILAI RMSE
    rms = np.mean(librosa.feature.rms(y=y))
    rms_median = np.median(librosa.feature.rms(y=y))
    rms_std_dev = np.std(librosa.feature.rms(y=y))
    rms_kurtosis = kurtosis(librosa.feature.rms(y=y)[0])
    rms_skew = skew(librosa.feature.rms(y=y)[0])

    # spectral centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Menggabungkan semua ciri-ciri menjadi sebuah list
    features = [zcr_mean, zcr_median, zcr_std_dev, zcr_kurtosis, zcr_skew,
                rms, rms_median, rms_std_dev, rms_kurtosis, rms_skew,
                spectral_centroid, spectral_bandwidth, spectral_rolloff]

    return features


columns = ['ZCR Mean', 'ZCR Median', 'ZCR Std Dev', 'ZCR Kurtosis', 'ZCR Skew',
           'RMSE', 'RMSE Median', 'RMSE Std Dev', 'RMSE Kurtosis', 'RMSE Skew',
           'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff']


def record_audio(duration=2):
    st.info("Recording audio... Click 'Stop Recording' after 2 seconds.")
    audio_data = sd.rec(int(44100 * duration),
                        samplerate=44100, channels=2, dtype='int16')
    sd.wait()
    st.success("Recording complete!")
    return audio_data

    
    # Convert sounddevice array to pydub AudioSegment
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=44100,
        sample_width=audio_data.dtype.itemsize,
        channels=2
    )
    
    st.success("Recording complete!")
    return audio_segment





def main():
    st.title("Audio Classification App")

    if st.button("Start Recording"):
        audio_data = record_audio()
        st.audio(audio_data.tobytes(), format="audio/wav",
                 start_time=0, sample_rate=44100)

        # Save recorded audio to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, audio_data, samplerate=44100)

        # Load the recorded audio file and process it
        y, sr = librosa.load(temp_wav.name, sr=None)
        statistics = calculate_statistics(y, sr)
        df = pd.DataFrame([statistics], columns=columns)
        new_data_df = pd.DataFrame(scaler.transform(df))

        # Perform prediction using the loaded model
        prediction = saved_model.predict(new_data_df)

        st.write("Prediction Result:", prediction)


if __name__ == "__main__":
    main()
