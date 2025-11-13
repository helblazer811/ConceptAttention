import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_concept_attention_video(concepts, concept_attention_maps, save_path, fps=4, color_map='inferno'):
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
                cmap=color_map, 
                interpolation='nearest',
                vmin=concept_attention_maps.min(),
                vmax=concept_attention_maps.max()
            )

    ani = animation.FuncAnimation(fig, update, frames=num_frames, repeat=False)
    ani.save(save_path, writer='ffmpeg', fps=fps)

def make_individual_videos(concepts, concept_attention_maps, save_dir, fps=4, color_map='inferno'):

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
                cmap=color_map, 
                interpolation='nearest',
                vmin=concept_attention_map.min(),
                vmax=concept_attention_map.max()
            )

        ani = animation.FuncAnimation(fig, update, frames=concept_attention_map.shape[0], repeat=False)
        ani.save(save_path, writer='ffmpeg', fps=fps)

    for i, concept in enumerate(concepts):
        concept_attention_map = concept_attention_maps[i]
        make_individual_video(concept, concept_attention_map, f"{save_dir}/{concept}_attention_video.mov")
