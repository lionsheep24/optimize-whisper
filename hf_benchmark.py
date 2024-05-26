from transformers import pipeline
import time
model_path:str = r"C:\Users\user\Documents\optimize-whisper\models\whisper-large-v2-hf"
# Whisper 모델을 위한 파이프라인 생성
whisper = pipeline("automatic-speech-recognition",model=model_path)

# 오디오 파일 불러오기
audio_file_path = "./test/test.wav"


# 추론 시작 전 시간 기록
start_time = time.time()

# 모델에 오디오 데이터 전달 및 텍스트로 변환
transcription = whisper(audio_file_path)

# 추론 종료 후 시간 기록
end_time = time.time()

# 결과 출력
print("Transcribed Text:", transcription["text"])
print("Inference Time:", end_time - start_time, "seconds")
