from pydub import AudioSegment
import io
import os
from google.cloud import speech_v1p1beta1 as speech

def split_audio(file_path, chunk_length_ms=10000):
    audio = AudioSegment.from_wav(file_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def transcribe_chunk(chunk, client):
    audio_content = chunk.raw_data
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US'
    )
    response = client.recognize(config=config, audio=audio)
    return response

def main():
    client = speech.SpeechClient()
    chunks = split_audio('C:\\Users\\acts7\\Downloads\\callout.wav')
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        response = transcribe_chunk(chunk, client)
        for result in response.results:
            print('Transcript: {}'.format(result.alternatives[0].transcript))

if __name__ == '__main__':
    main()
