import torch
from transformers import pipeline


from pytriton.decorators import batch
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor

DEVICE = "cuda:0"
NUM_MODEL_INSTANCE=2
BATCH_SIZE=16

class _InferFuncWrapper:
    def __init__(self, model):
        self._model = model

    @batch
    def __call__(self, **inputs):
        (input_batch,) = inputs.values()
        transcribed_text:str =self._model(input_batch)
        return [transcribed_text]

def _infer_function_factory(NUM_MODEL_INSTANCE):
    infer_fns = []
    for _ in NUM_MODEL_INSTANCE:
        transcriber = pipeline(model="openai/whisper-large-v2", device=DEVICE, batch_size=BATCH_SIZE)
        infer_fns.append(_InferFuncWrapper(model=transcriber))
    return infer_fns

if __name__ == "__main__":
    triton = Triton()
    triton.bind(
        model_name="Whisper",
        infer_func=_infer_function_factory,
        inputs=[Tensor(name="input", dtype=np.float16, shape=(-1,)),],
        outputs=[Tensor(name="output", dtype=bytes, shape=(-1,)),],
    )
    triton.run()