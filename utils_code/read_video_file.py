import subprocess
import os

def extract_frames(video_link, output_folder):
    
    """` ffmpeg -i path/video/sav_000001.mp4 -q:v 2 -start_number 0 path/folder_frame/'%05d.jpg' `"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    command = [
        'ffmpeg', '-i', video_link, '-q:v', '2', '-start_number', '0',
        os.path.join(output_folder, 'frame_%05d.jpg')
    ]
    subprocess.run(command, check=True)