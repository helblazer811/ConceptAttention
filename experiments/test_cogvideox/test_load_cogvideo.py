"""
    Test script for CogVideoX with concept attention using original code structure.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Add current directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modified_dit import ModifiedCogVideoXTransformer3DModel
from pipeline import ModifiedCogVideoXPipeline

# Import video_utils from concept_attention
from concept_attention.video_utils import make_concept_attention_video, make_individual_videos

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    # Load model - using CogVideoX-5b (can also use CogVideoX-2b for faster inference)
    model_id = "THUDM/CogVideoX-5b"
    dtype = torch.bfloat16

    print(f"Loading transformer from {model_id}...")
    transformer = ModifiedCogVideoXTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=dtype
    )

    print("Loading pipeline...")
    pipe = ModifiedCogVideoXPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=dtype
    ).to("cuda")

    # Define prompt and concepts to track
    prompt = "A golden retriever with a ball by a tree in the grass. Sky in background."
    concepts = ["dog", "grass", "sky", "tree", "ball"]

    print(f"Generating video for prompt: {prompt}")
    print(f"Tracking concepts: {concepts}")

    # Generate video with concept attention
    video, concept_attention_dict = pipe(
        prompt=prompt,
        concepts=concepts,
        num_videos_per_prompt=1,
        guidance_scale=6,
        num_frames=81,
        num_inference_steps=50,
        concept_attention_kwargs={
            "timesteps": list(range(0, 50)),
            "layers": list(range(0, 30)),
        }
    )
    video = video.frames[0]

    # Save the generated video
    output_video_path = os.path.join(OUTPUT_DIR, "output.mov")
    export_to_video(video, output_video_path, fps=8)
    print(f"Saved generated video to {output_video_path}")

    # Create concept attention visualization videos
    concept_attention_maps = concept_attention_dict["concept_attention_maps"]

    concept_attention_video_path = os.path.join(OUTPUT_DIR, "concept_attention.gif")
    make_concept_attention_video(concepts, concept_attention_maps, save_path=concept_attention_video_path, color_map="plasma")
    print(f"Saved concept attention video to {concept_attention_video_path}")

    make_individual_videos(concepts, concept_attention_maps, save_dir=OUTPUT_DIR, fps=8, color_map="plasma")
    print(f"Saved individual concept videos to {OUTPUT_DIR}")

    # Create cross attention visualization videos
    cross_attention_maps = concept_attention_dict["cross_attention_maps"]

    cross_attention_video_path = os.path.join(OUTPUT_DIR, "cross_attention.gif")
    make_concept_attention_video(concepts, cross_attention_maps, save_path=cross_attention_video_path, color_map="plasma")
    print(f"Saved cross attention video to {cross_attention_video_path}")

    cross_attention_dir = os.path.join(OUTPUT_DIR, "cross_attentions")
    os.makedirs(cross_attention_dir, exist_ok=True)
    make_individual_videos(concepts, cross_attention_maps, save_dir=cross_attention_dir, fps=8, color_map="plasma")
    print(f"Saved individual cross attention videos to {cross_attention_dir}")

    print("Done!")
