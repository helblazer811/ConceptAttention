
import torch
import torch.nn.functional as F
import einops
import matplotlib.pyplot as plt
import numpy as np

def overlay_heatmap_on_image(
    image,
    heatmap: torch.Tensor,
    save_path="results/heatmap_overlay.pdf",
):
    """
        Overlay the given heatmap on the image 
    """
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.to(torch.float32).detach().cpu().numpy()
    assert len(heatmap.shape) == 2, "Heatmap should be 2D"
    plt.figure()
    plt.imshow(image)
    # Upscale heatmap to image
    heatmap = F.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0), 
        size=image.shape[:2], 
        mode="bilinear", 
        align_corners=False
    )
    heatmap = heatmap.squeeze(0).squeeze(0).numpy()
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(save_path, dpi=300)

def plot_concept_heatmaps(
    image,
    concept_basis: torch.Tensor,
    concept_list: list[str],
    image_patch_vectors: torch.Tensor,
    softmax=True,
    normalize_maps=True
):
    """
        Plot the concept heatmaps to ensure that the concept basis is
        reasonable for the given image.  
    """
    assert len(image_patch_vectors.shape) in [4, 5], "Image patch vectors should be 4D or 5D, make sure you include layers and timesteps."
    fig, axs = plt.subplots(1, len(concept_list) + 1, figsize=(4 * len(concept_list) + 4, 4))
    # Normalize the concept basis
    # concept_basis = concept_basis / concept_basis.norm(dim=-1, keepdim=True)

    if len(image_patch_vectors.shape) == 5:
        image_patch_projections = einops.einsum(
            image_patch_vectors,
            concept_basis,
            "layers time heads patches d, layers time heads concepts d -> layers time heads concepts patches",
        )
        if softmax:
            image_patch_projections = torch.softmax(image_patch_projections, dim=-2)
        image_patch_projections = einops.reduce(
            image_patch_projections, 
            "layers time heads concepts patches -> concepts patches", 
            reduction="mean"
        )
        image_patch_projections = einops.rearrange(
            image_patch_projections, 
            "concepts (h w) -> concepts h w", 
            h=64, 
            w=64
        )
    else:
        image_patch_projections = einops.einsum(
            image_patch_vectors,
            concept_basis,
            "layers time patches d, layers time concepts d -> layers time concepts patches",
        )
        if softmax:
            image_patch_projections = torch.softmax(image_patch_projections, dim=-2)

        image_patch_projections = einops.reduce(
            image_patch_projections, 
            "layers time concepts patches -> concepts patches", 
            reduction="mean"
        )
        image_patch_projections = einops.rearrange(
            image_patch_projections, 
            "concepts (w h) -> concepts w h", 
            h=64, 
            w=64
        )
    image_patch_projections = image_patch_projections.to(torch.float32).detach().cpu().numpy()
    # Get min and max values
    min_val = image_patch_projections.min()
    max_val = image_patch_projections.max()

    if len(concept_list) > 30:
        for concept in concept_list:
            plt.figure()
            if normalize_maps:
                plt.imshow(
                    image_patch_projections[concept_list.index(concept)],
                    cmap="plasma",
                    vmin=min_val,
                    vmax=max_val
                )
            else:
                plt.imshow(
                    image_patch_projections[concept_list.index(concept)],
                    cmap="plasma"
                )
            plt.title(concept)
            plt.savefig(f"results/concept_heatmaps/{concept}.png")
            plt.close()
    else:
        # Plot the image
        axs[0].imshow(image)
        axs[0].set_title("Image")
        axs[0].axis("off")
        # Plot the concept heatmaps
        for i, concept in enumerate(concept_list):
            if normalize_maps:
                axs[i + 1].imshow(
                    image_patch_projections[i],
                    cmap="plasma",
                    vmin=min_val,
                    vmax=max_val
                )
            else:
                axs[i + 1].imshow(
                    image_patch_projections[i],
                    cmap="plasma"
                )
            axs[i + 1].set_title(concept)
            axs[i + 1].axis("off")
        # Save the figure
        plt.savefig("results/concept_heatmaps.png")
        plt.close()

def plot_coefficients_heatmap(
    coefficients: torch.Tensor,
    concepts: list[str],
    save_path="results/group_coding_heatmaps.png"
):
    # Convert the coefficients to a dictionary
    coefficients = coefficients.detach().cpu().numpy()
    coefficients = coefficients.T
    dictionaries = []
    for i in range(coefficients.shape[0]):
        dictionary = {}
        for j, concept in enumerate(concepts):
            dictionary[concept] = coefficients[i, j]
        dictionaries.append(dictionary)
    # Convert dictionaries to numpy arrays
    dictionaries = [np.array([dictionary[concept] for concept in concepts]) for dictionary in dictionaries]
    dictionaries = np.stack(dictionaries, axis=0)
    dictionaries = einops.rearrange(
        dictionaries, 
        "(w h) concepts -> concepts w h", 
        w=64, 
        h=64
    )
    # Get min and max
    min_val = dictionaries.min()
    max_val = dictionaries.max()
    # Plot the coeffients of each dictioanry for each patch
    fig, axs = plt.subplots(1, len(concepts), figsize=(4 * len(concepts), 4))
    for concept_index, concept in enumerate(concepts):
        axs[concept_index].imshow(
            dictionaries[concept_index],
            cmap="plasma",
            # vmin=min_val,
            # vmax=max_val
        )
        axs[concept_index].set_title(concept)
        axs[concept_index].set_xticks([])
        axs[concept_index].set_yticks([])
        axs[concept_index].axis("off")

    plt.savefig(save_path)
    plt.close()