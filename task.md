+ Run example video
    + Path: /home/labelling/Project/segment-anything-2/data/test_video
    + ffmpeg -i /home/labelling/Project/segment-anything-2/sav_dataset/example/sav_000001.mp4 -q:v 2 -start_number 0 /home/labelling/Project/segment-anything-2/data/test_video/'%05d.jpg'

# Problem
+ what is the input from FE , video(must run ffmpeg) or folder image of video.
+ Response output for cvat -> show and export format data to training
+ Timeout when infer long video.
+ How to deploy service with triton or nuclio