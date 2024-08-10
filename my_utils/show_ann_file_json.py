import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

def visualize_coco_annotations(image_path, annotation_path):
    # Load the COCO annotations
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    coco = COCO(annotation_path)

    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the annotations for the image
    # image_id = coco.getImgIds(imgIds=[coco_data['images'][0]['id']])[0]
    # ann_ids = coco.getAnnIds(imgIds=[image_id])
    # anns = coco.loadAnns(ann_ids)
    anns = []
    anns.append(coco_data['annotations'][0])
    anns.append(coco_data['annotations'][1])

    # Draw the segmentation masks
    for ann in anns:
        ann['segmentation']['size'] = [image.shape[0], image.shape[1]]
        rle = ann['segmentation']
        rle= maskUtils.frPyObjects(rle, rle['size'][0], rle['size'][1])
        mask = maskUtils.decode(rle)
        color = np.random.randint(0, 255, 3).tolist()
        image[mask == 1] = image[mask == 1] * 0.5 + np.array(color) * 0.5

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Example usage
# image_path = 'path/to/your/image.jpg'
image_path = '/home/labelling/Project/segment-anything-2/Data_import_test/images/000001.jpg'
annotation_path = '/home/labelling/Project/segment-anything-2/Data_import_test/annotations/instances_test.json'
visualize_coco_annotations(image_path, annotation_path)