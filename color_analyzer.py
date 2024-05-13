from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:0"

# Too big to store in RAM
processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b") 
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
).to(DEVICE)

# Create inputs
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What do we see in this image?"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "In this image, we can see the city of New York, and more specifically the Statue of Liberty.",
            },
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "And how about this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    text=prompt,
    images=[
        load_image("/home/nathgoh/Github/color-analysis/IMG_1477.jpg"),
        load_image("/home/nathgoh/Github/color-analysis/IMG_1477.jpg"),
    ],
    return_tensors="pt",
)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}


# Generate
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)
