+ Run example video
    + Path: /home/labelling/Project/segment-anything-2/data/test_video
    + ffmpeg -i /home/labelling/Project/segment-anything-2/sav_dataset/example/sav_000001.mp4 -q:v 2 -start_number 0 /home/labelling/Project/segment-anything-2/data/test_video/'%05d.jpg'

# Problem
+ what is the input from FE , video(must run ffmpeg) or folder image of video.
+ Response output for cvat -> show and export format data to training
+ Timeout when infer long video.
+ How to deploy service with triton or nuclio

# Format COCO
+ https://github.com/cvat-ai/cvat?tab=readme-ov-file
+ https://cocodataset.org/#format-data

+ Export data to coco format
+ Import data after format to cvat and test result.

# Fix bug Error box after format to RLE
+ label by hand the same two object in model view result
+ using 

# How to run video with SAM2 and export to json file
+ Wrire function:  Read video by ffmpeg, save to folder, file name ...
+ get folder run 