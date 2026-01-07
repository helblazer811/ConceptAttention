# ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/helblazer811/ConceptAttention)
[![arxiv badge](https://img.shields.io/badge/arXiv-2502.04320-red)](https://arxiv.org/abs/2502.04320)

ConceptAttention is an interpretability method for multi-modal diffusion transformers. We implement it for the Flux DiT architecture in PyTorch. Check out the paper [here](https://arxiv.org/abs/2502.04320).

<p align="center">
    <img src="images/teaser.png" alt="Teaser Image" width="800"/>
</p>

# Code setup

You will then need to install the code here locally by running
```bash
pip install -e .
```

# Running the model 

Here is an example of how to run Flux with Concept Attention

```python
from concept_attention import ConceptAttentionFluxPipeline

pipeline = ConceptAttentionFluxPipeline(
    model_name="flux-schnell",
    device="cuda:0"
)

prompt = "A dragon standing on a rock. "
concepts = ["dragon", "rock", "sky", "cloud"]

pipeline_output = pipeline.generate_image(
    prompt=prompt,
    concepts=concepts,
    width=1024,
    height=1024,
)

image = pipeline_output.image
concept_heatmaps = pipeline_output.concept_heatmaps

image.save("image.png")
for concept, concept_heatmap in zip(concepts, concept_heatmaps):
    concept_heatmap.save(f"{concept}.png")
```

# Examples

Example scripts are located in the [`examples/`](examples) directory:

| Script | Description |
|--------|-------------|
| `encode_image_flux.py` | Encode an existing image and generate concept heatmaps (Flux 1) |
| `generate_image_flux.py` | Generate a new image with concept heatmaps (Flux 1) |
| `generate_image_flux2.py` | Generate a new image with concept heatmaps (Flux 2) |
| `encode_image_sd3.py` | Encode an existing image and generate concept heatmaps (SD3) |
| `generate_image_sd3.py` | Generate a new image with concept heatmaps (SD3) |
| `generate_video_cogvideox.py` | Generate a video with concept attention heatmaps (CogVideoX) |

To run an example:
```bash
cd examples
python generate_image_flux.py
```

Output images will be saved to `examples/results/flux/` or `examples/results/flux2/` depending on the model.

# Concept Attention Also Works on Video!

ConceptAttention can also be applied to video generation models. Here's an example using CogVideoX:

```python
from concept_attention.cogvideox import ModifiedCogVideoXTransformer3DModel, ModifiedCogVideoXPipeline
from diffusers.utils import export_to_video

# Load model
model_id = "THUDM/CogVideoX-5b"
transformer = ModifiedCogVideoXTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
pipe = ModifiedCogVideoXPipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.bfloat16
).to("cuda")

# Generate video with concept attention
prompt = "A golden retriever with a ball by a tree in the grass."
concepts = ["dog", "grass", "sky", "tree", "ball"]

video, concept_attention_dict = pipe(
    prompt=prompt,
    concepts=concepts,
    num_frames=81,
    num_inference_steps=50,
    concept_attention_kwargs={
        "timesteps": list(range(0, 50)),
        "layers": list(range(0, 30)),
    }
)

# Save video
export_to_video(video.frames[0], "output.mov", fps=8)

# Access concept attention maps (shape: num_concepts, num_frames, height, width)
concept_attention_maps = concept_attention_dict["concept_attention_maps"]
```

See the full example at [`examples/generate_video_cogvideox.py`](examples/generate_video_cogvideox.py). 

# Experiments

Each of our experiments are in separate directories in [`/experiments`](experiments). 

You can run one for example like this
```bash
cd experiments/qualitative_baseline_comparison
python generate_image.py # Generates test image using flux
python plot_flux_concept_attention.py # Generates concept attention maps and saves them in results. 
```

# Data Setup
To use ImageNetSegmentation you will need to download `gtsegs_ijcv.mat` into [`/experiments/imagenet_segmentation/data`](experiments/imagenet_segmentation/data/). 

```bash
cd experiments/imagenet_segmentation/data
wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
```


# Bibtex

```
@misc{helbling2025conceptattentiondiffusiontransformerslearn,
    title={ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features}, 
    author={Alec Helbling and Tuna Han Salih Meral and Ben Hoover and Pinar Yanardag and Duen Horng Chau},
    year={2025},
    eprint={2502.04320},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2502.04320}, 
}
```