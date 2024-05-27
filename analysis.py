import cv2 as cv

from utils.detection_utils import (
    detect_face,
    detect_skin
)

# model_id = "xtuner/llava-phi-3-mini-hf"

# prompt = "<|user|>\n<image>\nBased on the portrait, what color is the person's facial skin tone? Give me a hexadecimal equivalent value for the color.<|end|>\n<|assistant|>\n"
# image_file = "/home/nathgoh/Github/color-analysis/test_img.JPG"

# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     torch_dtype=torch.float16,
#     tie_weights=True,
#     device_map="auto"
# )

# processor = AutoProcessor.from_pretrained(model_id)


# raw_image = Image.open(image_file)
# inputs = processor(prompt, raw_image, return_tensors='pt').to(0, torch.float16)

# output = model.generate(**inputs, max_new_tokens=175, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))
face_cropped_img = detect_face(cv.imread("test_img_2.jpg"))
cv.imshow("face", face_cropped_img)
cv.waitKey(0)
# detected_skin_img = detect_skin(face_cropped_img)
# cv.imshow("skin", detected_skin_img)
# cv.waitKey(0)