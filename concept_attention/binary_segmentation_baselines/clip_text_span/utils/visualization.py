from PIL import Image

## Imports
from PIL import Image
from torchvision import transforms


def _convert_to_rgb(image):
    return image.convert("RGB")


visualization_preprocess = transforms.Compose(
    [
        transforms.Resize(size=224, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(224, 224)),
        _convert_to_rgb,
    ]
)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
