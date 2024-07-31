# Install
conda create --name segment_env python
conda activate segment_env
pip install -e .
pip install -e ".[demo]"

+ ffmpeg
+ `sudo apt install ffmpeg`

# Download model
cd checkpoints
bash download_ckpts.sh 

# Run video to frame
+ ` ffmpeg -i /home/oryza/Desktop/Projects/segment_tools/sav_dataset/example/sav_000001.mp4 -q:v 2 -start_number 0 /home/oryza/Desktop/Projects/segment_tools/checkpoints/data/test_video/'%05d.jpg' `