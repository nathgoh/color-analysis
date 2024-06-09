from typing import Tuple
import numpy as np
import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	pipeline,
)

from utils.color_extraction import (
	avg_k_means_high_freq_colors,
	k_means_color_clustering,
	highest_frequency_color,
	rgb_to_hex,
)
from utils.detection import (
	detect_face,
	detect_skin,
)

torch.random.manual_seed(0)


def extract_skintone_color(
	img: np.ndarray,
) -> Tuple[list, str]:
	"""
	Find the RGB and Hex color value for the skintone found from
	the portrait image.

	Args:
	    img (np.ndarray): Portrait image

	Returns:
	    Tuple[list, str]: Tuple with RGB values and Hex color equivalent
	"""
	face_cropped_img, face = detect_face(img)
	detected_skin_img = detect_skin(face_cropped_img)
	cluster_colors = k_means_color_clustering(detected_skin_img)
	highest_freq_color = highest_frequency_color(detected_skin_img)
	rgb = avg_k_means_high_freq_colors(highest_freq_color, cluster_colors)
	hex_color = rgb_to_hex(rgb[0], rgb[1], rgb[2])

	return rgb, hex_color


def analyze_skintone(hex_color: str) -> str:
	"""
	Determine the color palette based on the hex color value
	found for the skintone from the portrait image using LLM.

	Currently using Microsoft phi-3.

	Args:
	    hex_color (str): Hex color value of skintone found on portrait image

	Returns:
	    str: LLM's response
	"""

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
		"temperature": 0.8,
		"do_sample": True,
	}

	output = pipe(messages, **generation_args)

	return output[0]["generated_text"]
