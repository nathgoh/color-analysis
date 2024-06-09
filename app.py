import numpy as np
import gradio as gr

from analysis import extract_skintone_color, analyze_skintone


def portrait_processing(portrait):
	_, hex_color = extract_skintone_color(portrait)
	response = analyze_skintone(hex_color)

	return response


with gr.Blocks() as app:
	gr.Markdown(
		"""
        # Color Analysis Using LLM        
        ### Simply upload a portrait image in natural lighting, preferably with a emtpy background!
        We are using the Microsoft Phi-3 model for this application. 
        """
	)
	input_image = gr.Image(sources=["upload", "clipboard"])
	output_text = gr.Textbox(label="Response")
 
	b1 = gr.Button("Analyze")
	b1.click(portrait_processing, inputs=input_image, outputs=output_text)


if __name__ == "__main__":
	app.launch()
