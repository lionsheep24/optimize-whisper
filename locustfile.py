from locust import User, task, between, constant
import numpy as np
import soundfile
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import time

""" 오디오 파일을 로드하고 Triton 서버로 전송합니다. """


class TritonUser(User):
    wait_time = constant(0.001)  # 사용자가 작업 사이에 0.5초에서 3초 사이를 기다리게 설정

    def on_start(self):
        """ 사용자가 필요로 하는 자원을 초기화합니다. """
        self.server_url = "localhost:10200"  # Triton 서버 주소와 포트를 업데이트하세요
        self.model_name = "whisper-large-v2-tensorrt-llm"
        
        self.client = grpcclient.InferenceServerClient(url=self.server_url)

    @task
    def send_whisper(self):
        audio_path = "./test/test.wav"  # 경로를 업데이트하세요
        waveform, sample_rate = soundfile.read(audio_path)
        assert sample_rate == 16000, f"16k 샘플 레이트만 지원되지만 {sample_rate}을 받았습니다."
        duration = int(len(waveform) / sample_rate)
        # 10초 간격으로 패딩
        padding_duration = 30
        samples = np.zeros(
            (
                1,
                padding_duration * sample_rate * ((duration // padding_duration) + 1),
            ),
            dtype=np.float32,
        )
        samples[0, :len(waveform)] = waveform
        print(samples.shape)
        inputs = [
            grpcclient.InferInput("WAV", samples.shape, np_to_triton_dtype(samples.dtype)),
            grpcclient.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(samples)
        input_data_numpy = np.array(["<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>"], dtype=object).reshape((1, 1))
        inputs[1].set_data_from_numpy(input_data_numpy)

        outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTS")]

        start_time = time.time()
        try:
            response = self.client.infer(
                self.model_name,
                inputs,
                request_id="1",
                outputs=outputs
            )
            total_time = int((time.time() - start_time) * 1000)  # ms 단위로 변환
            decoding_results = response.as_numpy("TRANSCRIPTS")[0]
            print(decoding_results.decode("utf-8"))
            self.environment.events.request.fire(
                request_type="GRPC",
                name="send_whisper",
                response_time=total_time,
                response_length=0,
                exception=None,
            )
        except Exception as e:
            print(e)
            # total_time = int((time.time() - start_time) * 1000)  # ms 단위로 변환
            # self.environment.events.request.fire(
            #     request_type="GRPC",
            #     name="send_whisper",
            #     response_time=total_time,
            #     response_length=0,
            #     exception=e,
            # )

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")