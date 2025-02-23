import spaces
import gradio as gr
from PIL import Image
import math
import io
import base64

from concept_attention import ConceptAttentionFluxPipeline

IMG_SIZE = 210
COLUMNS = 5

EXAMPLES = [
    [
        "A dog by a tree",  # prompt
        "tree, dog, grass, background",  # words
        42,  # seed
    ],
    # [
    #     "A dragon",  # prompt
    #     "dragon, sky, rock, cloud",  # words
    #     42,  # seed
    # ],
    # [
    #     "A hot air balloon",  # prompt
    #     "balloon, sky, water, tree",  # words
    #     42,  # seed
    # ]
]

def update_default_concepts(prompt):
    default_concepts = {
        "A dog by a tree": ["dog", "grass", "tree", "background"],
        "A dragon": ["dragon", "sky", "rock", "cloud"],
        "A hot air balloon": ["balloon", "sky", "water", "tree"]
    }

    return gr.update(value=default_concepts.get(prompt, []))

pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda", offload_model=True)

def convert_pil_to_bytes(img):
    img = img.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

@spaces.GPU(duration=60)
def process_inputs(prompt, concepts, seed, layer_start_index, timestep_start_index):
    # print("Processing inputs")
    # assert layer_start_index is not None
    # assert timestep_start_index is not None

    if not prompt.strip():
        raise gr.exceptions.InputError("prompt", "Please enter a prompt")

    prompt = prompt.strip()

    print(concepts)
    # if not word_list.strip():
    #     gr.exceptions.InputError("words", "Please enter comma-separated words")

    # concepts = [w.strip() for w in word_list.split(",")]

    if len(concepts) == 0:
        raise gr.exceptions.InputError("words", "Please enter at least 1 concept")
    
    if len(concepts) > 9:
        raise gr.exceptions.InputError("words", "Please enter at most 9 concepts")

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

    # heatmaps_and_labels = [(concept_heatmaps[concept_index], concepts[concept_index]) for concept_index in range(len(concepts))]
    # all_images_and_labels = [(output_image, "Generated Image")] + heatmaps_and_labels
    # num_rows = math.ceil(len(all_images_and_labels) / COLUMNS)

    return output_image, \
        gr.update(value=output_space_maps_and_labels, columns=len(output_space_maps_and_labels)), \
        gr.update(value=cross_attention_maps_and_labels, columns=len(cross_attention_maps_and_labels))

with gr.Blocks(
    css="""
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .title { text-align: center; margin-bottom: 10px; }
        .authors { text-align: center; margin-bottom: 10px; }
        .affiliations { text-align: center; color: #666; margin-bottom: 10px; }
        .abstract { text-align: center; margin-bottom: 40px; }
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
        }
        .input-column-label {}
        .gallery {
            # scrollbar-width: thin;
            # scrollbar-color: #27272A;
        }

        .run-button-column {
            width: 100px !important;
        }
    """
    # ,
    # elem_classes="container"
) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# ConceptAttention: Visualize Any Concepts in Your Generated Images", elem_classes="title")
        # gr.Markdown("### Alec Helbling¹, Tuna Meral², Ben Hoover¹³, Pinar Yanardag², Duen Horng (Polo) Chau¹", elem_classes="authors")
        # gr.Markdown("### ¹Georgia Tech · ²Virginia Tech · ³IBM Research", elem_classes="affiliations")
        gr.Markdown("## Interpret generative models with precise, high-quality heatmaps. Check out our paper [here](https://arxiv.org/abs/2502.04320).", elem_classes="abstract")

        with gr.Row(scale=1, equal_height=True):
            with gr.Column(scale=3, elem_classes="input-column"):
                gr.HTML(
                    "Write a Prompt",
                    elem_classes="input-column-label"
                )
                prompt = gr.Dropdown(
                    ["A dog by a tree", "A dragon", "A hot air balloon"], 
                    # label="Prompt", 
                    container=False,
                    # scale=3,
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
                    # scale=1,
                    elem_classes="input"
                )
            # prompt = gr.Textbox(
            #     label="Enter your prompt", 
            #     placeholder="Enter your prompt", 
            #     value=EXAMPLES[0][0],
            #     scale=4,
            #     # show_label=True,
            #     container=False
            #     # height="80px"
            # )
            # words = gr.Textbox(
            #     label="Enter a list of concepts (comma-separated)", 
            #     placeholder="Enter a list of concepts (comma-separated)", 
            #     value=EXAMPLES[0][1],
            #     scale=4,
            #     # show_label=True,
            #     container=False
            #     # height="80px"
            # )

        num_rows_state = gr.State(value=1)  # Initial number of rows

        # generated_image = gr.Image(label="Generated Image", elem_classes="input-image")
        # gallery = gr.Gallery(
        #     label="Generated images", 
        #     show_label=True, 
        #     # elem_id="gallery",
        #     columns=COLUMNS, 
        #     rows=1,
        #     # object_fit="contain", 
        #     height="auto",
        #     elem_classes="gallery"
        # )

        with gr.Row(elem_classes="gallery", scale=8):

            with gr.Column(scale=1):
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
                    elem_classes="gallery"
                )

                cross_attention_gallery = gr.Gallery(
                    label="Cross Attention", 
                    show_label=True, 
                    # columns=3, 
                    rows=1,
                    object_fit="contain", 
                    height="200px",
                    elem_classes="gallery"
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

        # gr.Examples(examples=EXAMPLES, inputs=[prompt, concepts, seed, layer_start_index, timestep_start_index], outputs=[gallery, num_rows_state], fn=process_inputs, cache_examples=False)
        # num_rows_state.change(
        #     fn=lambda rows: gr.Gallery.update(rows=int(rows)),
        #     inputs=[num_rows_state],
        #     outputs=[gallery]
        # )

        prompt.change(update_default_concepts, inputs=[prompt], outputs=[concepts])

        # Automatically process the first example on launch
        demo.load(
            process_inputs, 
            inputs=[prompt, concepts, seed, layer_start_index, timestep_start_index], 
            outputs=[generated_image, concept_attention_gallery, cross_attention_gallery]
        )

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
