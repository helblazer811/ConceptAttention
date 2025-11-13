"""
    Load the cogvideo model. 
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from modified_dit import ModifiedCogVideoXTransformer3DModel
from pipeline import ModifiedCogVideoXPipeline

from concept_attention.video.video_utils import make_concept_attention_video, make_individual_videos

if __name__ == "__main__":
    # model_id = "THUDM/CogVideoX-2b"
    model_id = "THUDM/CogVideoX-5b"
    # model_id = "THUDM/CogVideoX1.5-5B"
    dtype = torch.bfloat16
    transformer = ModifiedCogVideoXTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype
    )

    pipe = ModifiedCogVideoXPipeline.from_pretrained(
        model_id, 
        transformer=transformer,
        torch_dtype=dtype
    ).to("cuda")

    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    prompt = "A golden retriever with a ball by a tree in the grass. Sky in background. "
    
    concepts = ["dog", "grass", "sky", "tree", "ball"]
    video, concept_attention_dict = pipe(
        prompt=prompt,
        concepts=concepts,
        num_videos_per_prompt=1,
        guidance_scale=6,
        # use_dynamic_cfg=True, 
        num_frames=81,
        num_inference_steps=50,
        # num_frames=81,
        concept_attention_kwargs={
            "timesteps": list(range(0, 50)),
            "layers": list(range(0, 30)),
        }
    )
    video = video.frames[0]

    concept_attention_maps = concept_attention_dict["concept_attention_maps"]
    make_concept_attention_video(concepts, concept_attention_maps, save_path="results/concept_attention.mp4", color_map="plasma")

    export_to_video(video, "results/output.mov", fps=8)
    make_individual_videos(concepts, concept_attention_maps, save_dir="results", fps=8, color_map="plasma")

    # Now make videos for the cross attentions
    cross_attention_maps = concept_attention_dict["cross_attention_maps"]
    make_concept_attention_video(concepts, cross_attention_maps, save_path="results/cross_attention.mp4", color_map="plasma")
    make_individual_videos(concepts, cross_attention_maps, save_dir="results/cross_attentions", fps=8, color_map="plasma")