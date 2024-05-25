from google.cloud import speech_v1p1beta1 as speech
import io

# size limit is 10mb
downpath = r"C:\\Users\\acts7\\Downloads"
inputfilepath = downpath + "\\" + "call.m4a"
print("dbg01")
client = speech.SpeechClient()
print("dbg02")
# 오디오 파일 읽기
with io.open(inputfilepath, 'rb') as audio_file:
    content = audio_file.read()
print("dbg03")
audio = speech.RecognitionAudio(content=content)
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code='en-US'
)
print("dbg04")
response = client.recognize(config=config, audio=audio)
print("dbg05")
for result in response.results:
    print('Transcript: {}'.format(result.alternatives[0].transcript))
