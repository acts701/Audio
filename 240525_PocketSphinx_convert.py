import subprocess
import os
from pydub import AudioSegment
from pydub.utils import which
import speech_recognition as sr
import configparser

config = configparser.ConfigParser()
config.read('config.json')
downpath = config['PATH']['DOWNLOAD_FOLDER']

# FFmpeg 및 ffprobe 경로 설정
ffmpeg_path = downpath + "ffmpeg/bin/ffmpeg.exe"  # FFmpeg 실행 파일 경로
ffprobe_path = downpath + "ffmpeg/bin/ffprobe.exe"  # ffprobe 실행 파일 경로

# FFmpeg 및 ffprobe 경로 환경 변수에 추가
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
os.environ["PATH"] += os.pathsep + os.path.dirname(ffprobe_path)

# FFmpeg 및 ffprobe 경로 확인
try:
    subprocess.run([ffmpeg_path, "-version"], check=True)
    subprocess.run([ffprobe_path, "-version"], check=True)
    print("FFmpeg and ffprobe are correctly installed and accessible.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred: {e}")

# 경로 설정 확인
print(f"FFmpeg found: {which('ffmpeg')}")
print(f"ffprobe found: {which('ffprobe')}")


inputfilepath = downpath + "\\" + "call_kor.m4a"
outputfilepath = downpath + "\\" + "call_kor_out.wav"

AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

def convert_m4a_to_wav(m4a_file_path, wav_file_path):
    audio = AudioSegment.from_file(m4a_file_path, format="m4a")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)  # LINEAR16 설정
    audio.export(wav_file_path, format="wav")

def transcribe_audio(wav_file_path, chunk_length_ms=30000):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(wav_file_path)
    
    total_length_ms = len(audio)
    transcript = ""

    for i in range(0, total_length_ms, chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk.export("chunk.wav", format="wav")
        
        with sr.AudioFile("chunk.wav") as source:
            audio_data = recognizer.record(source)
        
        try:
            # PocketSphinx를 사용한 음성 인식 (오프라인)
            text = recognizer.recognize_sphinx(audio_data)
            transcript += text + " "
            print(f"Chunk {i // chunk_length_ms + 1} transcript: {text}")
        except sr.UnknownValueError:
            print(f"Chunk {i // chunk_length_ms + 1}: Sphinx could not understand audio")
        except sr.RequestError as e:
            print(f"Chunk {i // chunk_length_ms + 1}: Sphinx error; {e}")

    return transcript.strip()

# M4A 파일을 WAV 파일로 변환
convert_m4a_to_wav(inputfilepath, outputfilepath)

# 변환된 WAV 파일을 텍스트로 변환
full_transcript = transcribe_audio(outputfilepath)
print("Full Transcript:", full_transcript)
