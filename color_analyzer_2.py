import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "xtuner/llava-phi-3-mini-hf"

prompt = "<|user|>\n<image>\nWhat do we see in this image?<|end|>\n<|assistant|>\n"
image_file = "/home/nathgoh/Github/color-analysis/IMG_1477.jpg"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(model_id)


raw_image = Image.open(image_file)
inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))