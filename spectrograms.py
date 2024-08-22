import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def load_audio_files_from_folder(folder_path):
    audio_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav') or filename.endswith('.mp3'):
            audio_files.append(os.path.join(folder_path, filename))
    return audio_files

def extract_spectrogram(audio_path, n_mels=128, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S, sr, hop_length

def save_spectrogram_image(S, sr, hop_length, output_path, figsize=(10, 4), dpi=100):
    plt.figure(figsize=figsize, dpi=dpi)
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.axis('off')  # 축 제거
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()

male_audio_files_path = input('첫 번째 음성 파일 경로를 입력하세요: ')
female_audio_files_path = input('두 번째 음성 파일 경로를 입력하세요: ')
male_spectrograms_path = input('첫 번째 스펙트로그램 저장 경로를 입력하세요: ')
female_spectrograms_path = input('두 번째 스펙트로그램 저장 경로를 입력하세요: ')


male_audio_files = load_audio_files_from_folder(male_audio_files_path)
female_audio_files = load_audio_files_from_folder(female_audio_files_path)

os.makedirs(male_spectrograms_path, exist_ok=True)
os.makedirs(female_spectrograms_path, exist_ok=True)

# 남성 오디오 파일 처리 및 스펙트로그램 저장
for i, audio_file in enumerate(male_audio_files):
    spectrogram, sr, hop_length = extract_spectrogram(audio_file)
    output_path = os.path.join(male_spectrograms_path, f'{os.path.basename(audio_file)}.png')
    save_spectrogram_image(spectrogram, sr, hop_length, output_path)

# 여성 오디오 파일 처리 및 스펙트로그램 저장
for i, audio_file in enumerate(female_audio_files):
    spectrogram, sr, hop_length = extract_spectrogram(audio_file)
    output_path = os.path.join(female_spectrograms_path, f'{os.path.basename(audio_file)}.png')
    save_spectrogram_image(spectrogram, sr, hop_length, output_path)