import base64
import io

import spaces
import gradio as gr
from PIL import Image
import requests
import numpy as np
import PIL

from concept_attention import ConceptAttentionFluxPipeline

# concept_attention_default_args = {
#     "model_name": "flux-schnell",
#     "device": "cuda",
#     "layer_indices": list(range(10, 19)),
#     "timesteps": list(range(2, 4)),
#     "num_samples": 4,
#     "num_inference_steps": 4
# }
IMG_SIZE = 250

def download_image(url):
    return Image.open(io.BytesIO(requests.get(url).content))

EXAMPLES = [
    [
        "A dog by a tree",  # prompt
        download_image("https://github.com/helblazer811/ConceptAttention/blob/master/images/dog_by_tree.png?raw=true"),
        "tree, dog, grass, background",  # words
        42,  # seed
    ],
    [
        "A dragon",  # prompt
        download_image("https://github.com/helblazer811/ConceptAttention/blob/master/images/dragon_image.png?raw=true"),
        "dragon, sky, rock, cloud",  # words
        42,  # seed
    ],
       [
        "A hot air balloon",  # prompt
        download_image("https://github.com/helblazer811/ConceptAttention/blob/master/images/hot_air_balloon.png?raw=true"),
        "balloon, sky, water, tree",  # words
        42,  # seed
    ]
]

pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda")

@spaces.GPU(duration=60)
def process_inputs(prompt, input_image, word_list, seed, num_samples, layer_start_index, timestep_start_index):
    print("Processing inputs")
    prompt = prompt.strip()
    if not word_list.strip():
        return None, "Please enter comma-separated words"

    concepts = [w.strip() for w in word_list.split(",")]

    if input_image is not None:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)
            input_image = input_image.convert("RGB")
            input_image = input_image.resize((1024, 1024))
        elif isinstance(input_image, PIL.Image.Image):
            input_image = input_image.convert("RGB")
            input_image = input_image.resize((1024, 1024))

        pipeline_output = pipeline.encode_image(
            image=input_image,
            concepts=concepts,
            prompt=prompt,
            width=1024,
            height=1024,
            seed=seed,
            num_samples=num_samples,
            layer_indices=list(range(layer_start_index, 19)),
        )

    else:
        pipeline_output = pipeline.generate_image(
            prompt=prompt,
            concepts=concepts,
            width=1024,
            height=1024,
            seed=seed,
            timesteps=list(range(timestep_start_index, 4)),
            num_inference_steps=4,
            layer_indices=list(range(layer_start_index, 19)),
        )

    output_image = pipeline_output.image
    concept_heatmaps = pipeline_output.concept_heatmaps

    html_elements = []
    for concept, heatmap in zip(concepts, concept_heatmaps):
        img = heatmap.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        html = f"""
        <div style='text-align: center; margin: 5px; padding: 5px;  overflow-x: auto; white-space: nowrap;'>
            <h1 style='margin-bottom: 10px;'>{concept}</h1>
            <img src='data:image/png;base64,{img_str}' style='width: {IMG_SIZE}px; display: inline-block; height: {IMG_SIZE}px;'>
        </div>
        """
        html_elements.append(html)

    combined_html = "<div style='display: flex; flex-wrap: wrap; justify-content: center;'>" + "".join(html_elements) + "</div>"
    return output_image, combined_html, None # None fills input_image with None


with gr.Blocks(
    css="""
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .title { text-align: center; margin-bottom: 10px; }
    .authors { text-align: center; margin-bottom: 10px; }
    .affiliations { text-align: center; color: #666; margin-bottom: 10px; }
    .content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .section {  }
    .input-image { width: 100%; height: 200px; }
    .abstract { text-align: center; margin-bottom: 40px; }
"""
) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features", elem_classes="title")
        gr.Markdown("### Alec Helbling¹, Tuna Meral², Ben Hoover¹³, Pinar Yanardag², Duen Horng (Polo) Chau¹", elem_classes="authors")
        gr.Markdown("### ¹Georgia Tech · ²Virginia Tech · ³IBM Research", elem_classes="affiliations")
        gr.Markdown(
            """
                We introduce ConceptAttention, an approach to interpreting the intermediate representations of diffusion transformers. 
                The user just gives a list of textual concepts and ConceptAttention will produce a set of saliency maps depicting 
                the location and intensity of these concepts in generated images. Check out our paper: [here](https://arxiv.org/abs/2502.04320). 
            """,
            elem_classes="abstract"
        )

        with gr.Row(elem_classes="content"):
            with gr.Column(elem_classes="section"):
                gr.Markdown("### Input")
                prompt = gr.Textbox(label="Enter your prompt")
                words = gr.Textbox(label="Enter a list of concepts (comma-separated)")
                # gr.HTML("<div style='text-align: center;'> <h3> Or </h3> </div>")
                image_input = gr.Image(type="numpy", label="Upload image (optional)", elem_classes="input-image")
                # Set up advanced options 
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Seed", value=42)
                    num_samples = gr.Slider(minimum=1, maximum=10, step=1, label="Number of Samples", value=4)
                    layer_start_index = gr.Slider(minimum=0, maximum=18, step=1, label="Layer Start Index", value=10)
                    timestep_start_index = gr.Slider(minimum=0, maximum=4, step=1, label="Timestep Start Index", value=2)

            with gr.Column(elem_classes="section"):
                gr.Markdown("### Output")
                output_image = gr.Image(type="numpy", label="Output image")

        with gr.Row():
            submit_btn = gr.Button("Process")

        with gr.Row(elem_classes="section"):
            saliency_display = gr.HTML(label="Saliency Maps")

        submit_btn.click(
            fn=process_inputs, 
            inputs=[prompt, image_input, words, seed, num_samples, layer_start_index, timestep_start_index], outputs=[output_image, saliency_display, image_input]
        )
        # .then(
        #     fn=lambda component: gr.update(value=None),
        #     inputs=[image_input],
        #     outputs=[]
        # )

        gr.Examples(examples=EXAMPLES, inputs=[prompt, image_input, words, seed], outputs=[output_image, saliency_display], fn=process_inputs, cache_examples=False)

if __name__ == "__main__":
    demo.launch(max_threads=1)
    #     share=True,
    #     server_name="0.0.0.0",
    #     inbrowser=True,
    #     # share=False,
    #     server_port=6754,
    #     quiet=True,
    #     max_threads=1
    # )
