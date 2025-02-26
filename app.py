import spaces
import gradio as gr
from PIL import Image
import math
import io
import base64
import subprocess
import os

from concept_attention import ConceptAttentionFluxPipeline

IMG_SIZE = 210
COLUMNS = 5

def update_default_concepts(prompt):
    default_concepts = {
        "A dog by a tree": ["dog", "grass", "tree", "background"],
        "A man on the beach": ["man", "dirt", "ocean", "sky"],
        "A hot air balloon": ["balloon", "sky", "water", "tree"]
    }

    return gr.update(value=default_concepts.get(prompt, []))

pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell") # , offload_model=True) # , device="cuda:2", offload_model=True)

def convert_pil_to_bytes(img):
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

@spaces.GPU(duration=60)
def process_inputs(prompt, concepts, seed, layer_start_index, timestep_start_index):
    try:
        if not prompt:
            raise gr.Error("Please enter a prompt", duration=10)

        if not prompt.strip():
            raise gr.Error("Please enter a prompt", duration=10)

        prompt = prompt.strip()

        if len(concepts) == 0:
            raise gr.Error("Please enter at least 1 concept", duration=10)
        
        if len(concepts) > 9:
            raise gr.Error("Please enter at most 9 concepts", duration=10)

        pipeline_output = pipeline.generate_image(
            prompt=prompt,
            concepts=concepts,
            width=1024,
            height=1024,
            seed=seed,
            timesteps=list(range(timestep_start_index, 4)),
            num_inference_steps=4,
            layer_indices=list(range(layer_start_index, 19)),
            softmax=True if len(concepts) > 1 else False
        )

        output_image = pipeline_output.image

        output_space_heatmaps = pipeline_output.concept_heatmaps
        output_space_heatmaps = [heatmap.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST) for heatmap in output_space_heatmaps]
        output_space_maps_and_labels = [(output_space_heatmaps[concept_index], concepts[concept_index]) for concept_index in range(len(concepts))]

        cross_attention_heatmaps = pipeline_output.cross_attention_maps
        cross_attention_heatmaps = [heatmap.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST) for heatmap in cross_attention_heatmaps]
        cross_attention_maps_and_labels = [(cross_attention_heatmaps[concept_index], concepts[concept_index]) for concept_index in range(len(concepts))]

        return output_image, \
            gr.update(value=output_space_maps_and_labels, columns=len(output_space_maps_and_labels)), \
            gr.update(value=cross_attention_maps_and_labels, columns=len(cross_attention_maps_and_labels))

    except gr.Error as e:
        return None, gr.update(value=[], columns=1), gr.update(value=[], columns=1)

with gr.Blocks(
    css="""
        .container { 
            max-width: 1300px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .application {
            max-width: 1200px;
        }
        .generated-image {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%; /* Ensures full height */
        }

        .input {
            height: 47px;
        }
        .input-column {
            flex-direction: column;
            gap: 0px;
            height: 100%;
        }
        .input-column-label {}
        .gallery {
            height: 220px;
        }
        .run-button-column {
            width: 100px !important;
        }

        .gallery-container {
            scrollbar-width: thin;
            scrollbar-color: grey black;
        }
       
        /* Show only on screens wider than 768px (adjust as needed) 
        @media (min-width: 1024px) {
            .svg-container {
                min-width: 150px;
                width: 200px;
                padding-top: 540px;
            }
        }

        @media (min-width: 1280px) {
            .svg-container {
                min-width: 200px;
                width: 300px;
                padding-top: 420px;
            }
        }
         @media (min-width: 1530px) {
            .svg-container {
                min-width: 200px; 
                width: 300px;
                padding-top: 400px;
            }
        }

        */

        @media (min-width: 1024px) {
            .svg-container {
                min-width: 250px;
            }
            #concept-attention-callout-svg {
                width: 250px;
            }
        }


        @media (max-width: 1024px) {
            .svg-container {
                display: none !important;
            }
            #concept-attention-callout-svg {
                display: none;
            }
        }

        .header {
            display: flex;
            flex-direction: column;
        }
        #title {
            font-size: 4.4em;
            color: #F3B13E;
            text-align: center;
            margin: 5px;
        }
        #subtitle {
            font-size: 3.0em;
            color: #FAE2BA;
            text-align: center;
            margin: 5px;
        }
        #abstract { 
            text-align: center; 
            font-size: 2.0em;
            color:rgb(219, 219, 219);
            margin: 5px;
            margin-top: 10px;
        }
        #links {
            text-align: center;
            font-size: 2.0em;
            margin: 5px;
        }
        #links a {
            color: #93B7E9;
            text-decoration: none;
        }

        .svg-container {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .caption-label {
            font-size: 1.15em;
        }

        .gallery label {
            font-size: 1.15em;
        }

    """
) as demo:

    # with gr.Column(elem_classes="container"):


        with gr.Row(elem_classes="container", scale=8):
            
            with gr.Column(elem_classes="application-content", scale=10):

                with gr.Row(scale=3, elem_classes="header"):
                    gr.HTML("""
                        <h1 id='title'> ConceptAttention </h1>
                        <h1 id='subtitle'> Visualize Any Concepts in Your Generated Images </h1>
                        <h1 id='abstract'> Interpret diffusion models with precise, high-quality heatmaps. </h1>
                        <h1 id='links'> <a href='https://arxiv.org/abs/2502.04320'> Paper </a> | <a href='https://github.com/helblazer811/ConceptAttention'> Code </a> </h1>
                    """)

                with gr.Row(elem_classes="input-row", scale=2):
                    with gr.Column(scale=4, elem_classes="input-column", min_width=250):
                        gr.HTML(
                            "Write a Prompt",
                            elem_classes="input-column-label"
                        )
                        prompt = gr.Dropdown(
                            ["A dog by a tree", "A man on the beach", "A hot air balloon"], 
                            container=False,
                            allow_custom_value=True,
                            elem_classes="input"
                        )

                    with gr.Column(scale=7, elem_classes="input-column"):
                        gr.HTML(
                            "Select or Write Concepts",
                            elem_classes="input-column-label"
                        )
                        concepts = gr.Dropdown(
                            ["dog", "grass", "tree", "dragon", "sky", "rock", "cloud", "balloon", "water", "background"], 
                            value=["dog", "grass", "tree", "background"], 
                            multiselect=True, 
                            label="Concepts",
                            container=False,
                            allow_custom_value=True,
                            # scale=4,
                            elem_classes="input",
                            max_choices=5
                        )

                    with gr.Column(scale=1, min_width=100, elem_classes="input-column run-button-column"):
                        gr.HTML(
                            "&#8203;",
                            elem_classes="input-column-label"
                        )
                        submit_btn = gr.Button(
                            "Run",
                            elem_classes="input"
                        )

                with gr.Row(elem_classes="gallery-container", scale=8):

                    with gr.Column(scale=1, min_width=250):
                        generated_image = gr.Image(
                            elem_classes="generated-image",
                            show_label=False
                        )
                        
                    with gr.Column(scale=4):
                        concept_attention_gallery = gr.Gallery(
                            label="Concept Attention (Ours)", 
                            show_label=True, 
                            # columns=3, 
                            rows=1,
                            object_fit="contain", 
                            height="200px",
                            elem_classes="gallery",
                            elem_id="concept-attention-gallery",
                            # scale=4
                        )

                        cross_attention_gallery = gr.Gallery(
                            label="Cross Attention", 
                            show_label=True, 
                            # columns=3, 
                            rows=1,
                            object_fit="contain", 
                            height="200px",
                            elem_classes="gallery",
                            # scale=4
                        )

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Seed", value=42)
                    layer_start_index = gr.Slider(minimum=0, maximum=18, step=1, label="Layer Start Index", value=10)
                    timestep_start_index = gr.Slider(minimum=0, maximum=4, step=1, label="Timestep Start Index", value=2)

                submit_btn.click(
                    fn=process_inputs, 
                    inputs=[prompt, concepts, seed, layer_start_index, timestep_start_index], 
                    outputs=[generated_image, concept_attention_gallery, cross_attention_gallery]
                )

                prompt.change(update_default_concepts, inputs=[prompt], outputs=[concepts])

                # Automatically process the first example on launch
                demo.load(
                    process_inputs, 
                    inputs=[prompt, concepts, seed, layer_start_index, timestep_start_index], 
                    outputs=[generated_image, concept_attention_gallery, cross_attention_gallery]
                )

            with gr.Column(scale=2, min_width=200, elem_classes="svg-column"):

                with gr.Row(scale=8):
                    gr.HTML("<div></div>")

                with gr.Row(scale=4, elem_classes="svg-container"):
                    concept_attention_callout_svg = gr.HTML(
                        "<img src='/gradio_api/file=ConceptAttentionCallout.svg' id='concept-attention-callout-svg'/>",
                        # container=False,
                    )

                with gr.Row(scale=4):
                    gr.HTML("<div></div>")

if __name__ == "__main__":
    if os.path.exists("/data-nvme/zerogpu-offload"):
        subprocess.run("rm -rf /data-nvme/zerogpu-offload/*", env={}, shell=True)
    demo.launch(
        allowed_paths=["."]
    )
    #     share=True,
    #     server_name="0.0.0.0",
    #     inbrowser=True,
    #     # share=False,
    #     server_port=6754,
    #     quiet=True,
    #     max_threads=1
    # )
