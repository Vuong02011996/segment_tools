from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import shutil
import os
import re
import torch
import matplotlib
from utils_code.read_video_file import extract_frames
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import json
from utils_code.utils import binary_mask_to_rle, show_mask, show_points
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
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://vizo.tanika.ai"],  
    # allow_origins=["*"],  
    allow_credentials=True,
    # allow_methods=["*"],  
    allow_methods=["DELETE", "GET", "POST", "PUT"],
    allow_headers=["*"],  
)

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

    # Sort using the numeric part extracted from the filename
    # frame_names.sort(key=lambda p: int(re.search(r'\d+', p).group()))
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
    video_link = request.get("video_dir")
    auto_from_to = request.get("auto_from_to") # [0, 482]
    print("mul_points: ", mul_points)
    print("mul_labels: ", mul_labels)
    print("ann_obj_ids: ", ann_obj_ids)
    print("frame_idx: ", frame_idx)
    print("video_link: ", video_link)
    print("auto_from_to: ", auto_from_to)


    # frame_folder = '/home/labelling/Project/segment-anything-2/data/file_images'
    frame_folder_test = '/home/labelling/Project/segment-anything-2/data/test_frame_folders_test'
    extract_frames(video_link, frame_folder_test)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(frame_folder_test)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
    ]
    

    # Sort using the numeric part extracted from the filename
    frame_names.sort(key=lambda p: int(re.search(r'\d+', p).group()))
    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0][6:]))
    
    frame_names = frame_names[:10]

    # Define source and destination directories
    src_dir = frame_folder_test
    dst_dir = '/home/labelling/Project/segment-anything-2/data/test_frame_folders_test2'
    # Check if the directory exists
    if os.path.exists(dst_dir):
        # Delete the directory and its contents
        shutil.rmtree(dst_dir)
        print(f"Directory {dst_dir} has been deleted.")
    else:
        print(f"Directory {dst_dir} does not exist.")

    # Create destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Copy files and rename them
    print(frame_names)
    for filename in os.listdir(src_dir):
        if filename.startswith('frame_'):
            # print("filename: ", filename)
            if filename in frame_names:
                new_filename = filename.replace('frame_', '', 1)
                shutil.copy(os.path.join(src_dir, filename), os.path.join(dst_dir, new_filename))
    

    # SAM 2 requires stateful inference for interactive video segmentation, so we need to initialize an inference state on this video.
    inference_state = predictor.init_state(video_path=dst_dir)

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
        # image_path = os.path.join(video_dir, frame_name)
        image_id = frame_idx + 1
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

            # decompressed_rles, heights, widths = maskUtils.decompress([compressed_rle])
            # Calculate uncompressed_counts, area and bounding box manually C2
            rle = binary_mask_to_rle(mask_for_rle)
            # Create the COCO annotation
            annotation = {
                "id": ann_id,
                "image_id": image_id,
                "category_id": obj_id,
                "segmentation": rle,
                "area": int(maskUtils.area(compressed_rle)), # int(mask.sum())
                "bbox": maskUtils.toBbox(compressed_rle).tolist(),
                "iscrowd": 1,
                "attributes": {
                "occluded": False
                }
            }

            # Add the annotation to the list
            coco_annotations.append(annotation)
            ann_id += 1
    
            images.append({"license": 0, 
                       "id": image_id, 
                       "width": rle['size'][1], 
                       "height":  rle['size'][0], 
                       "file_name": frame_name,
                       "flickr_url":"",
                       "coco_url":"",
                       "date_captured":0
                       })
            
    # Create the COCO annotation file
    coco_data = {
                "licenses": [
                    {
                        "name": "",
                        "id": 0,
                        "url": ""
                    }
                ],
                "info": {
                    "contributor": "",
                    "date_created": "",
                    "description": "",
                    "url": "",
                    "version": "",
                    "year": ""
                },
                "categories": [],
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
    # data_import_path = "/home/labelling/Project/segment-anything-2/"
    # coco_file_path = os.path.join(data_import_path, "instances_test.json")
    # with open(coco_file_path, "w") as f:
    #     json.dump(coco_data, f)

    # return { "coco_file_path": coco_file_path}
    return coco_data



@app.get("/")
def read_root():
    return {"Service is running ..."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5959)