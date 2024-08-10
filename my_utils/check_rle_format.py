import numpy as np
from pycocotools import mask as maskUtils


"""
 The values alternate between the lengths of runs of background pixels (0s) and foreground pixels (1s).
 Normally, the counts should sum up to the total number of pixels in the mask (i.e., height * width).

0 0 1 1 0 0
0 0 1 1 0 0
0 0 0 0 0 0
1 1 1 1 1 0

The uncompressed RLE should logically be:

2 zeros (background)
2 ones (foreground)
2 zeros (background)
2 zeros (background)
2 ones (foreground)
2 zeros (background)
6 zeros (background)
5 ones (foreground)
1 zero (background)
[2, 2, 2, 2, 2, 2, 6, 5, 1]
"""

def mask_to_uncompressed_rle(mask):
    """
    Convert a binary mask to Uncompressed RLE using pycocotools.
    
    Parameters:
        mask (np.ndarray): Binary mask of shape (height, width) with 1s for object and 0s for background.
    
    Returns:
        dict: Uncompressed RLE in COCO format.
    """
    # Ensure mask is of type np.uint8 and in Fortran order
    mask = np.asfortranarray(mask.astype(np.uint8))


    # Use pycocotools to encode the mask
    rle = maskUtils.encode(mask)
    
    # # Extract counts
    # counts = rle['counts']
    
    # # If counts is a byte string, decode to get the run-lengths
    # if isinstance(counts, bytes):
    #     counts = counts.decode('utf-8')
    
    # # Convert the run-lengths string to a list of integers
    # uncompressed_counts = [int(i) for i in counts]

    
    # Use pycocotools to encode the mask
    rle = maskUtils.encode(mask)
    
    # The counts are in byte format; convert them to list of integers
    counts = rle['counts']

    uncompressed_counts = []
    prev_pixel = -1
    run_length = 0

     
    for i, count in enumerate(counts):
        if i % 2 == 0:  # count at even indices are background
            run_length += count
        else:  # count at odd indices are foreground
            uncompressed_counts.append(run_length)
            run_length = count
    
    
    return {"counts": uncompressed_counts, "size": [mask.shape[0], mask.shape[1]]}

# Example usage
# Create a binary mask (for demonstration purposes)
mask = np.array([
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0]
], dtype=np.uint8)

# Convert the mask to uncompressed RLE format
uncompressed_rle = mask_to_uncompressed_rle(mask)
print(uncompressed_rle)
