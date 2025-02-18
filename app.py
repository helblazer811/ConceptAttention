import spaces
import gradio as gr
from PIL import Image
import math

from concept_attention import ConceptAttentionFluxPipeline

IMG_SIZE = 250
COLUMNS = 5

EXAMPLES = [
    [
        "A dog by a tree",  # prompt
        "tree, dog, grass, background",  # words
        42,  # seed
    ],
    [
        "A dragon",  # prompt
        "dragon, sky, rock, cloud",  # words
        42,  # seed
    ],
    [
        "A hot air balloon",  # prompt
        "balloon, sky, water, tree",  # words
        42,  # seed
    ]
]

pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device="cuda")

@spaces.GPU(duration=60)
def process_inputs(prompt, word_list, seed, layer_start_index, timestep_start_index):
    print("Processing inputs")
    assert layer_start_index is not None
    assert timestep_start_index is not None

    prompt = prompt.strip()
    if not word_list.strip():
        gr.exceptions.InputError("words", "Please enter comma-separated words")

    concepts = [w.strip() for w in word_list.split(",")]

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
    concept_heatmaps = pipeline_output.concept_heatmaps
    concept_heatmaps = [heatmap.resize((IMG_SIZE, IMG_SIZE), resample=Image.NEAREST) for heatmap in concept_heatmaps]

    heatmaps_and_labels = [(concept_heatmaps[concept_index], concepts[concept_index]) for concept_index in range(len(concepts))]
    all_images_and_labels = [(output_image, "Generated Image")] + heatmaps_and_labels

    num_rows = math.ceil(len(all_images_and_labels) / COLUMNS)

    gallery = gr.Gallery(
        label="Generated images", 
        show_label=True, 
        columns=[COLUMNS], 
        rows=[num_rows],
        object_fit="contain"
    )

    return gallery

with gr.Blocks(
    css="""
    .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
    .title { text-align: center; margin-bottom: 10px; }
    .authors { text-align: center; margin-bottom: 10px; }
    .affiliations { text-align: center; color: #666; margin-bottom: 10px; }
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

        with gr.Row(scale=1):
            prompt = gr.Textbox(
                label="Enter your prompt", 
                placeholder="Enter your prompt", 
                value=EXAMPLES[0][0],
                scale=4,
                show_label=True,
                container=False
                # height="80px"
            )
            words = gr.Textbox(
                label="Enter a list of concepts (comma-separated)", 
                placeholder="Enter a list of concepts (comma-separated)", 
                value=EXAMPLES[0][1],
                scale=4,
                show_label=True,
                container=False
                # height="80px"
            )
            submit_btn = gr.Button(
                "Run",
                min_width="100px",
                scale=1
            )

        # generated_image = gr.Image(label="Generated Image", elem_classes="input-image")
        gallery = gr.Gallery(
            label="Generated images", 
            show_label=True, 
            # elem_id="gallery",
            columns=[COLUMNS], 
            rows=[1],
            object_fit="contain", 
            # height="auto"
        )
        with gr.Accordion("Advanced Settings", open=False):
            seed = gr.Slider(minimum=0, maximum=10000, step=1, label="Seed", value=42)
            layer_start_index = gr.Slider(minimum=0, maximum=18, step=1, label="Layer Start Index", value=10)
            timestep_start_index = gr.Slider(minimum=0, maximum=4, step=1, label="Timestep Start Index", value=2)


        submit_btn.click(
            fn=process_inputs, 
            inputs=[prompt, words, seed, layer_start_index, timestep_start_index], 
            outputs=[gallery]
        )

        gr.Examples(examples=EXAMPLES, inputs=[prompt, words, seed, layer_start_index, timestep_start_index], outputs=[gallery], fn=process_inputs, cache_examples=False)

        # Automatically process the first example on launch
        # demo.load(process_inputs, inputs=[prompt, words, seed, layer_start_index, timestep_start_index], outputs=[gallery])


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
