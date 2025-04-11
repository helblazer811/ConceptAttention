# ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/helblazer811/ConceptAttention)
[![arxiv badge](https://img.shields.io/badge/arXiv-2502.04320-red)](https://arxiv.org/abs/2502.04320)

ConceptAttention is an interpretability method for multi-modal diffusion transformers. We implement it for the Flux DiT architecture in PyTorch. Check out the paper [here](https://arxiv.org/abs/2502.04320).

<p align="center">
    <img src="teaser.png" alt="Teaser Image" width="800"/>
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