import cv2 as cv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.color_extraction import (
    avg_k_means_high_freq_colors,
    k_means_color_clustering,
    highest_frequency_color,
    rgb_to_hex,
)
from utils.detection import detect_face, detect_skin

torch.random.manual_seed(0)

img = cv.imread("test_img_2.jpg")
img = cv.resize(img, (2048, 2048))
face_cropped_img, face = detect_face(img)
detected_skin_img = detect_skin(face_cropped_img)
cluster_colors = k_means_color_clustering(detected_skin_img, 5)
highest_freq_color = highest_frequency_color(detected_skin_img)
rgb = avg_k_means_high_freq_colors(highest_freq_color, cluster_colors)
hex_color = rgb_to_hex(rgb[0], rgb[1], rgb[2])

content = f"""
    My skin tone color is {hex_color} in natural lighting. Can you tell me what color palette of 
    clothing would best suit me? (Deep autumn, winter, spring, etc.) and let me know how confident 
    you are about it.
"""

messages = [
    {
        "role": "user",
        "content": content,
    }
]

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]["generated_text"])
