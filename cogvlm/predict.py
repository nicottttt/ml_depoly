import asyncio
import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from cog import BasePredictor, Input, Path, current_scope, ConcatenateIterator, File
from threading import Thread
import requests
import json

MODEL_NAME = "/path_to_model/cogvlm2-llama3-chat-19B"

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            trust_remote_code=True,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

    async def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Input prompt", default="Describe this image."),
        top_p: float = Input(
            description="Sample from top p probability mass",
            ge=0.0,
            le=1.0,
            default=0.9,
        ),
        temperature: float = Input(
            description="Sampling temperature",
            default=0.7,
            ge=0.0,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate",
            default=2048,
            ge=0,
        ),
    ) -> ConcatenateIterator[str]:  
        image_input = Image.open(str(image)).convert('RGB')
        
        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=prompt,
            images=[image_input],
            template_version='chat'
        )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(self.device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(self.device),
            "images": (
                [[input_by_model["images"][0].to(self.device).to(torch.bfloat16)]]
                if image is not None
                else None
            ),
        }

        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "do_sample": True,
        }

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for new_token in streamer:
            if "<|im_end|>" in new_token or "\n\n" in new_token:
                break
            yield new_token
        thread.join()  
