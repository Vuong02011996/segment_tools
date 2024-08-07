from fastapi import FastAPI
import uvicorn
import numpy as np
import os
import cv2
import torch
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt

from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

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


"""
List api:
+ given a positive point and image, return mask of objects (plus show mask to image and return image)
+ given a negative point and image, return mask of objects (plus show mask to image and return image)
+ 

"""
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

    # show the results on the current (interacted) frame on all objects
    segmented_image_path = f"{video_dir}/mul_segmented_images"
    os.makedirs(segmented_image_path, exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.title(f"frame {ann_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
    show_points(points, labels, plt.gca())
    for i, out_obj_id in enumerate(out_obj_ids):
        show_points(*prompts[out_obj_id], plt.gca())
        show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)
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