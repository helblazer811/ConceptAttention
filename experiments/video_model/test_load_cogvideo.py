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

def make_concept_attention_video(concepts, concept_attention_maps, fps=4):
    """
        For each concept, create a video using matplotlib where each frame is displayed as a heatmap.

        Inputs:
            concepts: List[str]
            concept_attention_maps: torch.Tensor of shape (num_concepts, num_frames, height, width)
    """
    num_concepts, num_frames, height, width = concept_attention_maps.shape

    fig, axes = plt.subplots(1, num_concepts, figsize=(len(concepts) * 7, 5))
    if num_concepts == 1:
        axes = [axes]

    def update(frame):
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_title(concepts[i])
            heatmap = concept_attention_maps[i, frame, :, :].to(torch.float32).cpu().numpy()
            ax.imshow(
                heatmap, 
                cmap='inferno', 
                interpolation='nearest',
                vmin=concept_attention_maps.min(),
                vmax=concept_attention_maps.max()
            )

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save('results/concept_attention_video.mov', writer='ffmpeg', fps=fps)

def make_individual_videos(concepts, concept_attention_maps, fps=4):

    def make_individual_video(concept, concept_attention_map, save_path):

        h, w = concept_attention_map.shape[1:]  # Height and width of the attention map
        aspect_ratio = w / h
        fig, ax = plt.subplots(figsize=(5 * aspect_ratio, 5))
        fig.patch.set_visible(False)  # Remove background

        def update(frame):
            ax.clear()
            ax.axis('off')  # Turn off axis ticks and labels
            ax.set_xticks([])  # Remove x ticks
            ax.set_yticks([])  # Remove y ticks
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # Remove padding
            # ax.set_title(concept)
            heatmap = concept_attention_map[frame, :, :].to(torch.float32).cpu().numpy()
            ax.imshow(
                heatmap, 
                cmap='inferno', 
                interpolation='nearest',
                vmin=concept_attention_map.min(),
                vmax=concept_attention_map.max()
            )

        ani = animation.FuncAnimation(fig, update, frames=concept_attention_map.shape[0], repeat=False)
        ani.save(save_path, writer='ffmpeg', fps=fps)

    for i, concept in enumerate(concepts):
        concept_attention_map = concept_attention_maps[i]
        make_individual_video(concept, concept_attention_map, f"results/{concept}_attention_video.mov")

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

    prompt = "A golden retriever dog running in a grassy park. Trees in the background. Blue sky. Sun in the sky. The dog looks very happy and cheerful. "
    
    concepts = ["dog", "grass", "sky", "tree"]
    video, concept_attention_dict = pipe(
        prompt=prompt, 
        concepts=concepts,
        num_videos_per_prompt=1,
        guidance_scale=6, 
        # use_dynamic_cfg=True, 
        num_frames=81,
        num_inference_steps=50,
        # num_frames=81,
        concept_attention_kwargs= {
            "timesteps": list(range(0, 50)),
            "layers": list(range(0, 30)),
        }
    )
    video = video.frames[0]

    print(concept_attention_dict["concept_attention_maps"].shape)

    concept_attention_maps = concept_attention_dict["concept_attention_maps"]
    make_concept_attention_video(concepts, concept_attention_maps)

    export_to_video(video, "results/output.mov", fps=8)

    make_individual_videos(concepts, concept_attention_maps, fps=8)