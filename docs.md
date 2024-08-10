# Export output segmentation to coco format
`
info{
"year": int, "version": str, "description": str, "contributor": str, "url": str, "date_created": datetime,
}

images{
"id": int, "width": int, "height": int, "file_name": str, "license": int, "flickr_url": str, "coco_url": str, "date_captured": datetime,
}

licenses{
"id": int, "name": str, "url": str,
}
annotations{
"id": int, "image_id": int, "category_id": int, "segmentation": RLE or [polygon], "area": float, "bbox": [x,y,width,height], "iscrowd": 0 or 1,
}

categories[{
"id": int, "name": str, "supercategory": str,
}]
`

## anotation
`
 annotation = {
                "image_id": image_id,
                "category_id": obj_id,
                "segmentation": {
                    "counts": uncompressed_counts,
                    "size": list(mask.shape)
                },
                "area": int(mask.sum()),
                "bbox": maskUtils.toBbox(rle).tolist(),
                "iscrowd": 1 

            }
` 

## RLE( Run-Length Encoding), uncompress RLE and polygon in segmentation
+ Concept: polygon, compress RLE, uncompress RLE format: https://github.com/pytorch/vision/issues/8351
+ `iscrowd` = 0 (polygon): [[x1, y1, x2, y2],[x1,y1,x2,y2],...], where x, y are the coordinates of vertices
+ `iscrowd` = 1 (uncompressed RLE):
+ The compact/encoded/compressed RLE format: {"size", [height, width], "counts": str}

+ two forms of RLE used in COCO
    + uncompressed RLE: 
        + frPyObjects    - Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.
        + https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx#L119
    + Compressed RLE:
        + 


# Export / Import coco format to CVAT
+ https://github.com/cvat-ai/cvat/blob/develop/site/content/en/docs/manual/advanced/formats/format-coco.md
+ Convert RLE: 
https://github.com/cvat-ai/cvat/blob/develop/cvat/apps/dataset_manager/formats/transformations.py#L40


+ Convert to RLE format => convert again uncompress RLE format
    https://github.com/cocodataset/cocoapi/issues/386

# Using polygon format
+ Code export json COCO format example but using polygon: 
https://github.com/Mortyzhang/Mask2polygon_tool/blob/main/Mask2polygon.py