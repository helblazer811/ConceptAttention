import base64
import io

import spaces
import gradio as gr
from PIL import Image

from concept_attention import ConceptAttentionFluxPipeline

concept_attention_default_args = {
    "model_name": "flux-schnell",
    "device": "cuda",
    "layer_indices": list(range(10, 19)),
    "timesteps": list(range(4)),
    "num_samples": 4,
    "num_inference_steps": 4
}
IMG_SIZE = 250

EXAMPLES = [
    [
        "A fluffy cat sitting on a windowsill",  # prompt
        "cat.jpg",  # image
        "fur, whiskers, eyes",  # words
        42,  # seed
    ],
    # ["Mountain landscape with lake", "cat.jpg", "sky, trees, water", 123],
    # ["Portrait of a young woman", "monkey.png", "face, hair, eyes", 456],
]


pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda")


@spaces.GPU(duration=60)
def process_inputs(prompt, input_image, word_list, seed):
    prompt = prompt.strip()
    if not word_list.strip():
        return None, "Please enter comma-separated words"

    concepts = [w.strip() for w in word_list.split(",")]

    if input_image is not None:
        input_image = Image.fromarray(input_image)
        input_image = input_image.convert("RGB")
        input_image = input_image.resize((1024, 1024))

        pipeline_output = pipeline.encode_image(
            image=input_image,
            concepts=concepts,
            prompt=prompt,
            width=1024,
            height=1024,
            seed=seed,
            num_samples=concept_attention_default_args["num_samples"]
        )
    else:
        pipeline_output = pipeline.generate_image(
            prompt=prompt,
            concepts=concepts,
            width=1024,
            height=1024,
            seed=seed,
            timesteps=concept_attention_default_args["timesteps"],
            num_inference_steps=concept_attention_default_args["num_inference_steps"],
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
    return output_image, combined_html


with gr.Blocks(
    css="""
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .title { text-align: center; margin-bottom: 10px; }
    .authors { text-align: center; margin-bottom: 20px; }
    .affiliations { text-align: center; color: #666; margin-bottom: 40px; }
    .content { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .section { border: 2px solid #ddd; border-radius: 10px; padding: 20px; }
"""
) as demo:
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
                gr.HTML("<div style='text-align: center;'> <h1> Or </h1> </div>")
                image_input = gr.Image(type="numpy", label="Upload image (optional)")

            with gr.Column(elem_classes="section"):
                gr.Markdown("### Output")
                output_image = gr.Image(type="numpy", label="Output image")

        with gr.Row():
            submit_btn = gr.Button("Process")

        with gr.Row(elem_classes="section"):
            saliency_display = gr.HTML(label="Saliency Maps")

        submit_btn.click(
            fn=process_inputs, 
            inputs=[prompt, image_input, words, seed], outputs=[output_image, saliency_display]
        )

        gr.Examples(examples=EXAMPLES, inputs=[prompt, image_input, words, seed], outputs=[output_image, saliency_display], fn=process_inputs, cache_examples=False)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        inbrowser=True,
        # share=False,
        server_port=6754,
        quiet=True,
        max_threads=1
    )
