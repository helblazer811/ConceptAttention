import gradio as gr
import numpy as np
from PIL import Image
import io
import base64
import torch
from skimage.transform import resize

EXAMPLES = [
    [
        "A fluffy cat sitting on a windowsill",  # prompt
        "cat.jpg",                               # image
        "fur, whiskers, eyes",                   # words
        42                                       # seed
    ],
    [
        "Mountain landscape with lake",
        "cat.png",
        "sky, trees, water",
        123
    ],
    [
        "Portrait of a young woman",
        "monkey.png",
        "face, hair, eyes",
        456
    ]
]


def process_inputs(prompt, input_image, word_list, seed):
    if not word_list.strip():
        return None, "Please enter comma-separated words"
    
    np.random.seed(seed)
    words = [w.strip() for w in word_list.split(',')]
    output_image = resize(input_image, (512,512)) if input_image is not None else np.random.rand(512, 512, 3)
    
    html_elements = []
    for word in words:
        smap = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        img = Image.fromarray(smap)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        html = f"""
        <div style='text-align: center; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>
            <h3 style='margin-bottom: 10px;'>{word}</h3>
            <img src='data:image/png;base64,{img_str}' style='width: 128px; height: 128px;'>
        </div>
        """
        html_elements.append(html)
    
    combined_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center;'>" + "".join(html_elements) + "</div>"
    return output_image, combined_html

with gr.Blocks(css="""
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .title { text-align: center; margin-bottom: 10px; }
    .authors { text-align: center; margin-bottom: 20px; }
    .affiliations { text-align: center; color: #666; margin-bottom: 40px; }
    .content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .section { border: 2px solid #ddd; border-radius: 10px; padding: 20px; }
""") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features", elem_classes="title")
        gr.Markdown("**Alec Helbling**¹, **Tuna Meral**², **Ben Hoover**¹³, **Pinar Yanardag**², **Duen Horng (Polo) Chau**¹", elem_classes="authors")
        gr.Markdown("¹Georgia Tech · ²Virginia Tech · ³IBM Research", elem_classes="affiliations")
        
        with gr.Row(elem_classes="content"):
            with gr.Column(elem_classes="section"):
                gr.Markdown("### Input")
                prompt = gr.Textbox(label="Enter your prompt")
                words = gr.Textbox(label="Enter words (comma-separated)")
                seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Seed", value=42)
                
            
            with gr.Column(elem_classes="section"):
                gr.Markdown("### Output")
                image_input = gr.Image(type="numpy", label="Upload image (optional)")
                
        with gr.Row():
            submit_btn = gr.Button("Process")

        with gr.Row(elem_classes="section"):
            saliency_display = gr.HTML(label="Saliency Maps")
        
        submit_btn.click(
            fn=process_inputs,
            inputs=[prompt, image_input, words, seed],
            outputs=[image_input, saliency_display]
        )

        gr.Examples(
            examples=EXAMPLES,
            inputs=[prompt, image_input, words, seed],
            outputs=[image_input, saliency_display],
            fn=process_inputs,
            cache_examples=False
        )

if __name__ == "__main__":
    demo.launch()