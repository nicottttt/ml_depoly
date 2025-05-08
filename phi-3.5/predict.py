from cog import BasePredictor, Input, ConcatenateIterator
import os
import time
import torch
import subprocess
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import TextIteratorStreamer

MODEL_NAME = "./Phi-3-medium-128k-instruct"
MESSAGES="""<|system|>
{sys_prompt}<|end|>
<|user|>
{user_prompt}<|end|>
<|assistant|>
"""

class Predictor(BasePredictor):
    def setup(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
        else :
            self.device = "cpu"
        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True
        )

    def predict(
        self,
        prompt: str = Input(description="请输入发送给模型的提示词"),
        max_length: int = Input(
            description="生成回答的最大token数量, 一个单词大概是2-3个tokens",
            ge=1,
            le=4096,
            default=200,
        ),
        temperature: float = Input(
            description="调整随机性, 大于1是随机的, 0是确定的.",
            ge=0.1,
            le=5.0,
            default=0.1,
        ),
        top_p: float = Input(
            description="解码文本时,从最可能的标记的前百分之p中采样, 数值越低忽略更少的token",
            ge=0.01,
            le=1.0,
            default=1.0,
        ),
        top_k: int = Input(
            description="解码文本时, 来自前 k 个最有可能的标记的样本, 数值越低忽略更少的token",
            default=1,
        ),
        repetition_penalty: float = Input(
            description="对生成文本中重复的单词进行处罚;1 表示无惩罚，大于 1 的值不鼓励重复，小于 1 的值鼓励重复。",
            ge=0.01,
            le=10.0,
            default=1.1,
        ),
        system_prompt: str = Input(
            description="System prompt.",
            default="AI助手"
        ),
        seed: int = Input(
            description="随机数生成器的种子", default=None
        ),
    ) -> ConcatenateIterator[str]:
        if seed is None:
            seed = torch.randint(0, 100000, (1,)).item()
        torch.random.manual_seed(seed)
        chat_format = MESSAGES.format(sys_prompt=system_prompt, user_prompt=prompt)
        tokens = self.tokenizer(chat_format, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, remove_start_token=True)
        input_ids = tokens.input_ids.to(device=self.device)
        max_length = input_ids.shape[1] + max_length
        generation_kwargs = dict(
            input_ids=input_ids,
            max_length=max_length,
            return_dict_in_generate=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
            do_sample=True
        )
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for _, new_text in enumerate(streamer):
            yield new_text
        thread.join()
