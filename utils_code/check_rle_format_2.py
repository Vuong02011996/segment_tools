import torch
from pycocotools import mask
import matplotlib.pyplot as plt

segmentation = {
    'size': [200, 200],
    'counts': '\\`>>c5<F9I:F<Eg0YOo0QO8H7J3L5L2N2N2O1N2O1N2O1O1O001O000000O10001N100O2O0O2N2N1O2M3N3M2M3N3K5K5I7G9C>^Oa0E<F;G9F>@b\\>'
}

def segmentation_to_mask_2(segmentation, *, canvas_size):    
    try:
        segmentation = (
            mask.frPyObjects(segmentation, *canvas_size)
            if isinstance(segmentation, dict)
            else mask.merge(mask.frPyObjects(segmentation, *canvas_size))
        )
        return torch.from_numpy(mask.decode(segmentation))
    except:
        return torch.from_numpy(mask.decode(segmentation))
    else:
        print("Masks has to be in either one of the form: polygons, uncompressed RLE, or compressed RLE")


def segmentation_to_mask_3(segmentation, *, canvas_size):
    if isinstance(segmentation["counts"], str):
        # If segmentation is already in RLE format, no need to process further
        pass
    elif isinstance(segmentation, dict):
        # Convert uncompressed RLE to encoded RLE mask
        segmentation= mask.frPyObjects(segmentation, *canvas_size)
    else: 
        # Convert polygons to encoded RLE mask
        segmentation = mask.merge(mask.frPyObjects(segmentation, *canvas_size))

    return torch.from_numpy(mask.decode(segmentation))

# Create a figure and axes
fig, axes = plt.subplots(1, 3, figsize=(10, 5))

# Show actual mask
axes[0].imshow(mask.decode(segmentation))
axes[0].set_title('Real segmentation')
axes[0].axis('off')

# Show output of suggested fix
axes[1].imshow(segmentation_to_mask_2(segmentation, canvas_size=(200,200)))
axes[1].set_title('Output_2')
axes[1].axis('off')

# Show output of suggested fix
axes[2].imshow(segmentation_to_mask_3(segmentation, canvas_size=(200,200)))
axes[2].set_title('Output_3')
axes[2].axis('off')

# Adjust layout
plt.tight_layout()

# Show the images
plt.show()