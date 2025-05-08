import os, random, asyncio, time
from typing import AsyncIterator, List, Union
from cog import BasePredictor, Input, ConcatenateIterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL_NAME = "./Mistral-Small-24B-Instruct-2501"
PROMPT_TEMPLATE = "<s>[INST] {prompt} [/INST] "
SYSTEM_PROMPT = "你是一个AI助手"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

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
        self.tokenizer.pad_token = self.tokenizer.eos_token

    async def predict(
        self,
        prompt: str = Input(description="模型提示词输入"),
        system_prompt: str = Input(
            description="模型的系统提示，有助于指导模型行为。",
            default=SYSTEM_PROMPT,
        ),
        max_tokens: int = Input(
            description="最大生成tokens数量", ge=0, le=4096,default=512
        ),
        top_p: float = Input(description="Top P", default=0.95),
        top_k: int = Input(description="Top K", default=10),
        min_p: float = Input(description="Min P", default=0),
        typical_p: float = Input(description="Typical P", default=1.0),
        tfs: float = Input(description="Tail-Free Sampling", default=1.0),
        frequency_penalty: float = Input(
            description="Frequency penalty", ge=0.0, le=2.0, default=0.0
        ),
        presence_penalty: float = Input(
            description="Presence penalty", ge=0.0, le=2.0, default=0.0
        ),
        repeat_penalty: float = Input(
            description="Repetition penalty", ge=0.0, le=2.0, default=1.1
        ),
        temperature: float = Input(description="Temperature", default=0.8),
        seed: int = Input(description="Seed", default=None),
    ) -> ConcatenateIterator[str]:
        full_prompt = f"{system_prompt}\n\n{prompt}\n\n答："

        if seed:
                set_seed(seed)
                print(f"Retrieved seed: {seed}")

        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            padding=True 
        ).to(self.model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
        thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": input_ids,
                "pad_token_id": self.tokenizer.eos_token_id,
                "attention_mask": attention_mask,
                "do_sample": True,
                "max_new_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "typical_p": typical_p,
                "repetition_penalty": repeat_penalty,
                "temperature": temperature,
                "streamer": streamer,
            },
        )

        thread.start()

        for new_token in streamer:
            if "<|im_end|>" in new_token or "\n\n" in new_token:
                break
            yield new_token
        thread.join()      