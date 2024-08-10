from fastapi import FastAPI
import uvicorn
import numpy as np
import os
import cv2
import torch
import matplotlib

from my_utils.check_rle_format_2 import segmentation_to_mask_2, segmentation_to_mask_3
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import json
from my_utils.utils import binary_mask_to_rle, binary_mask_to_uncompressed_rle, calculate_area_and_bbox, convert_rle_to_list, decompress_rle_string, show_mask, show_mask_binary, show_masks_comparison, show_points
# https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools
from pycocotools import mask as maskUtils

# check cuda and init model
# use bfloat16 for the entire notebook
# torch.cuda.set_device(1)
torch.autocast(device_type="cuda:1", dtype=torch.bfloat16).__enter__()
torch.cuda.empty_cache()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("torch.backends.cudnn.allow_tf32: ", torch.backends.cudnn.allow_tf32)

sam2_checkpoint = "/home/labelling/Project/segment-anything-2/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
image_test_path = "/home/labelling/Project/segment-anything-2/data/"

app = FastAPI()


@app.post("/segment_one_object")
def segment_one_object(request: dict):
    
    # Process the request JSON and extract the necessary data
    points = request.get("points")
    labels = request.get("labels")
    ann_obj_id = request.get("ann_obj_id")

    frame_idx = request.get("frame_idx")
    video_dir = request.get("video_dir")
    print("points: ", points)
    print("labels: ", labels)
    print("ann_obj_id: ", ann_obj_id)
    print("frame_idx: ", frame_idx)
    print("video_dir: ", video_dir)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
    inference_state = predictor.init_state(video_path=video_dir)

    # Perform segmentation on the image using the given point
    ann_frame_idx = frame_idx  # the frame index we interact with
    ann_obj_id = ann_obj_id  # give a unique id to each object we interact with (it can be any integers)

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array(points, dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(labels, np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # Your segmentation code goes here
    segmented_image_path = f"{video_dir}/segmented_images"
    os.makedirs(segmented_image_path, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
    # Save the masked image
    result_image_with_point = f"{segmented_image_path}/result_{frame_names[ann_frame_idx]}"
    plt.savefig(result_image_with_point)

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    
    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

        # Save the masked image
        path_img_masked = f"{segmented_image_path}/result_video_{frame_names[out_frame_idx]}"
        plt.savefig(path_img_masked)

    return {"result_image_have_point": result_image_with_point, "path_result": segmented_image_path}



@app.post("/segment_multiple_objects")
def segment_multiple_objects(request: dict):
    
    # Process the request JSON and extract the necessary data
    mul_points = request.get("points")
    mul_labels = request.get("labels")
    ann_obj_ids = request.get("ann_obj_ids")

    frame_idx = request.get("frame_idx")
    video_dir = request.get("video_dir")
    print("mul_points: ", mul_points)
    print("mul_labels: ", mul_labels)
    print("ann_obj_ids: ", ann_obj_ids)
    print("frame_idx: ", frame_idx)
    print("video_dir: ", video_dir)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
    inference_state = predictor.init_state(video_path=video_dir)

    prompts = {}
    for i , ann_obj_id in enumerate(ann_obj_ids):
        # Perform segmentation on the image using the given point
        ann_frame_idx = frame_idx
        points = np.array(mul_points[i], dtype=np.float32)
        labels = np.array(mul_labels[i], np.int32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        prompts[ann_obj_id] = points, labels

        # Check output model
        image_masked = (out_mask_logits[0]).cpu().numpy()
        print("image_masked: ", image_masked.shape)
        image_masked = image_masked > 0.0

 
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # Convert the segmentation masks to COCO format
    images = []
    coco_annotations = []
    for frame_idx, frame_name in enumerate(frame_names):
        image_path = os.path.join(video_dir, frame_name)
        image_id = frame_idx + 1

        # Load the image
        image = Image.open(image_path)
        width, height = image.size
        width, height = width+1, height+1
        images.append({"license": 0, 
                       "id": image_id, 
                       "width": width, 
                       "height": height, 
                       "file_name": frame_name,
                       "flickr_url":"",
                       "coco_url":"",
                       "date_captured":0
                       })


        # Get the segmentation masks for this frame
        frame_masks = video_segments[frame_idx]

        # Convert each mask to COCO format
       
        ann_id = 1
        for obj_id, mask in frame_masks.items():
            mask_binary = np.array(mask, dtype=np.uint8)
            mask_binary = np.where(mask_binary > 0, 1, 0)

            mask_for_rle = np.asfortranarray(mask_binary.astype(np.uint8))
            mask_for_rle = np.squeeze(mask_for_rle)


            # Convert the binary mask to RLE format
            compressed_rle = maskUtils.encode(mask_for_rle)
            # number_list = convert_rle_to_list(compressed_rle["counts"])

                        # Create a figure and axes
            fig, axes = plt.subplots(1, 3, figsize=(10, 5))

            # Show actual mask
            axes[0].imshow(maskUtils.decode(compressed_rle))
            axes[0].set_title('Real segmentation')
            axes[0].axis('off')

            # Show output of suggested fix
            axes[1].imshow(segmentation_to_mask_2(compressed_rle, canvas_size=(height,width)))
            axes[1].set_title('Output_2')
            axes[1].axis('off')

            # # Show output of suggested fix
            # axes[2].imshow(segmentation_to_mask_3(compressed_rle, canvas_size=(height,width)))
            # axes[2].set_title('Output_3')
            # axes[2].axis('off')

            # Adjust layout
            plt.tight_layout()

            # Show the images
            path_img_masked = f"{image_test_path}test_format_2.png"
            plt.savefig(path_img_masked)

            decompressed_rles, heights, widths = maskUtils.decompress([compressed_rle])

            # Calculate uncompressed_counts, area and bounding box manually C1
            uncompressed_counts = binary_mask_to_uncompressed_rle(mask_binary)
            area, bbox = calculate_area_and_bbox(mask_binary)

            # Calculate uncompressed_counts, area and bounding box manually C2
            rle = binary_mask_to_rle(mask_for_rle)
            # Create the COCO annotation
            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": obj_id,
                "segmentation": rle,
                # "segmentation": {
                #     # "counts": uncompressed_counts,
                #     "counts": decompressed_rles[0],
                #     # "size": [mask.shape[1], mask.shape[2]] # height, width of the mask
                #     "size": [height, width] # height, width of the mask
                # },
                "area": int(maskUtils.area(compressed_rle)), # int(mask.sum())
                "bbox": maskUtils.toBbox(compressed_rle).tolist(),
                # "area": area, # int(mask.sum())
                # "bbox": bbox,
                "iscrowd": 1,
                "attributes": {
                "occluded": False
                }
            }

            # Add the annotation to the list
            coco_annotations.append(annotation)
            ann_id += 1

    # Create the COCO annotation file
    coco_data = {
                "info": {"year": 2021, "version": "2021", "description": "zjx", "contributor": "zjx", "url": "",
                        "date_created": "2021.07.06"},
                "categories": [],
                "license": {"id": 1, "url": "", "name": "zhangjiaxin"},
                "images": images,
                "annotations": coco_annotations
            }
    
    # Add the image information to the COCO data
    # Add categories to the COCO data
    # class_index = {1: "building"}
    # for s, k in enumerate(list(class_index.keys())):
    #     coco_data["categories"].append({"id": k, "name": class_index[k], "supercategory": "building"})

    categories = [
        {
            "id": 1,
            "name": "Edge",
            "supercategory": ""
        },
        {
            "id": 2,
            "name": "Mobility",
            "supercategory": ""
        },
        {
            "id": 3,
            "name": "Obstacle",
            "supercategory": ""
        },
        {
            "id": 4,
            "name": "Insecure-zone",
            "supercategory": ""
        },
        {
            "id": 5,
            "name": "free-space",
            "supercategory": ""
        }
    ]
    coco_data["categories"] = categories
    
    # Save the COCO annotation file
    data_import_path = "/home/labelling/Project/segment-anything-2/"
    coco_file_path = os.path.join(data_import_path, "instances_test.json")
    with open(coco_file_path, "w") as f:
        json.dump(coco_data, f)

    return { "coco_file_path": coco_file_path}




@app.get("/")
def read_root():
    return {"Service is running ..."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5959)

    """
    Output models:
    out_mask_logits: the predicted mask logits for each object, auto appended with each new point added in loop
        + out_mask_logits[i] shape [1, 811, 1444](c, h, w) is the mask logits for the i-th object
        + < 0.0
    """