import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from itertools import groupby

def show_mask_binary(mask_binary):
    # Ensure mask_binary is a 2D numpy array
    if len(mask_binary.shape) == 3 and mask_binary.shape[0] == 1:
        mask_binary = np.squeeze(mask_binary, axis=0)

    # Create a colored background (e.g., black)
    background_color = [0, 0, 0]  # RGB for black
    background = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
    background[:] = background_color

    # Create an image with white where mask == 1
    mask_color = [255, 255, 255]  # RGB for white
    mask_image = np.zeros_like(background)
    mask_image[mask_binary == 1] = mask_color

    # Combine the background and mask
    combined_image = np.where(mask_binary[:, :, None] == 1, mask_image, background)

    # Display the combined image using Matplotlib
    plt.imshow(combined_image)
    plt.title('Mask with Background')
    plt.axis('off')  # Hide the axis
    plt.show()

def convert_to_mask_binary(mask):
    # Ensure mask has shape (1, 811, 1444)
    if mask.shape[0] == 1:
        mask_binary = np.squeeze(mask, axis=0)
    else:
        raise ValueError("The mask does not have the expected shape (1, 811, 1444)")
    
    return mask_binary

def show_masks_comparison(mask):
    # Convert mask to mask_binary
    mask_binary = convert_to_mask_binary(mask)
    
    # Plot the original mask and mask_binary side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original mask
    axes[0].imshow(mask[0], cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')  # Hide the axis
    
    # Binary mask
    axes[1].imshow(mask_binary, cmap='gray')
    axes[1].set_title('Binary Mask')
    axes[1].axis('off')  # Hide the axis
    
    plt.show()

def decompress_rle_string(rle_string, height, width):
    """
    Decompresses a compressed RLE string to an uncompressed RLE format.

    Args:
        rle_string (str): The compressed RLE string.
        height (int): The height of the image/mask.
        width (int): The width of the image/mask.

    Returns:
        list: The uncompressed RLE as a list of integers.
    """
    import re

    # Extract counts from the RLE string
    counts = [int(x) for x in re.findall(r'\d+', rle_string)]
    uncompressed_rle = []
    current_value = 0

    for count in counts:
        uncompressed_rle.extend([current_value] * count)
        current_value = 1 - current_value  # Toggle between 0 and 1

    return uncompressed_rle

def convert_rle_to_list(compressed_rle):
    # Step 1: Decode the bytes to a string
    decoded_string = compressed_rle.decode('utf-8')
    
    # Step 2: Split the string into individual components
    string_components = decoded_string.split()
    
    # Step 3: Convert each component to an integer
    number_list = [int(component) for component in string_components]
    
    return number_list

def show_mask_binary(mask_binary):
    if len(mask_binary.shape) == 3 and mask_binary.shape[0] == 1:
        mask_binary = np.squeeze(mask_binary, axis=0)

    # Create a colored background (e.g., black)
    background_color = [0, 0, 0]  # RGB for black
    background = np.zeros((mask_binary.shape[0], mask_binary.shape[1], 3), dtype=np.uint8)
    background[:] = background_color

    # Create an image with white where mask == 1
    mask_color = [255, 255, 255]  # RGB for white
    mask_image = np.zeros_like(background)
    mask_image[mask_binary == 1] = mask_color

    # Combine the background and mask
    combined_image = np.where(mask_binary[:, :, None] == 1, mask_image, background)
    return combined_image


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)  


def show_test():
    print("Test successful!")
       # # show the results on the current (interacted) frame on all objects
    # segmented_image_path = f"{video_dir}/mul_segmented_images"
    # os.makedirs(segmented_image_path, exist_ok=True)
    # plt.figure(figsize=(12, 8))
    # plt.title(f"frame {ann_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    # show_points(points, labels, plt.gca())
    # for i, out_obj_id in enumerate(out_obj_ids):
    #     show_points(*prompts[out_obj_id], plt.gca())
    #     show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
    #    # Save the masked image
    # result_image_with_point = f"{segmented_image_path}/result_{frame_names[ann_frame_idx]}"
    # plt.savefig(result_image_with_point)



    # # render the segmentation results every few frames
    # vis_frame_stride = 1
    # plt.close("all")
    # for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    #     plt.figure(figsize=(6, 4))
    #     plt.title(f"frame {out_frame_idx}")
    #     plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    
    #     for out_obj_id, out_mask in video_segments[out_frame_idx].items():
    #         show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    #     # Save the masked image
    #     path_img_masked = f"{segmented_image_path}/result_video_{frame_names[out_frame_idx]}"
    #     plt.savefig(path_img_masked)


     # show_masks_comparison(mask)
    # Convert the mask to binary format
    # plt.figure(figsize=(6, 4))
    # plt.title(f"frame {out_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    # show_mask(mask, plt.gca(), obj_id=obj_id)
    # path_img_masked = f"{image_test_path}test.png"
    # plt.savefig(path_img_masked)


        # plt.figure(figsize=(6, 4))
    # plt.title(f"frame {out_frame_idx}")
    # plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    # show_mask(mask_binary, plt.gca(), obj_id=obj_id)
    # path_img_masked = f"{image_test_path}test2.png"
    # plt.savefig(path_img_masked)


        # # uncompressed_rle = decompress_rle_string(compressed_rle['counts'].decode('utf-8'), height, width)
    # # Step 1: Decompress the RLE to get the binary mask
    # binary_mask = maskUtils.decode(compressed_rle)

    # # Display the combined image using Matplotlib
    # combined_image = show_mask_binary(mask_binary)
    # plt.imshow(combined_image)
    # plt.title('Mask with Background')
    # plt.axis('off')  # Hide the axis
    # path_img_masked = f"{image_test_path}test4.png"
    # plt.savefig(path_img_masked)

        # Convert the binary mask to uncompressed RLE format
        # Squeeze the extra dimension

    # show_mask_binary(mask_binary)

    # combined_image = show_mask_binary(mask_binary)
    # # Display the combined image using Matplotlib
    # plt.imshow(combined_image)
    # plt.title('Mask with Background')
    # plt.axis('off')  # Hide the axis
    # path_img_masked = f"{image_test_path}test3.png"
    # plt.savefig(path_img_masked)

