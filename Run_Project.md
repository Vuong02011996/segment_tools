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


# Deploy cvat source 
git clone https://github.com/dx-tech-ai/labelling-cvat.git
cd labelling-cvat
docker compose down || docker compose -f docker-compose.yml -f docker-compose.dev.yml down
export CVAT_HOST=127.0.0.1

 sudo docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d


sudo docker exec -it cvat_server bash -ic 'python3 ~/manage.py c
reatesuperuser'
    + admin
    + vanvuong0440@gmail.com
    + pass: 123

## Error
+ Error response from daemon: Conflict. The container name "/nuclio" is already in use by container "ffe67c9dce0968e70c7171054e634bc3a20c5c7adf600b959ae77bbd6de38bff"
    + docker stop ffe
    + docker rm ffe

+ labelling is not in the sudoers file.  This incident will be reported
    + https://stackoverflow.com/questions/47806576/username-is-not-in-the-sudoers-file-this-incident-will-be-reported
    + In user dxtech: open file `sudo nano /etc/sudoers` add to file: `labelling ALL=(ALL)  ALL`

# Deyploy with dev
+ https://docs.cvat.ai/docs/contributing/development-environment/
+ Error:
    + Could not find library geos_c or load any of its variants
        + https://stackoverflow.com/questions/19742406/could-not-find-library-geos-c-or-load-any-of-its-variants
        + sudo apt-get install libgeos-dev



# Other
+ List service with port: `lsof -i :5959`
+ kill service with pid : kill -9 pid