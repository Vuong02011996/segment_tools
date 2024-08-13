# Setup env with SAM2
```
conda create --name segment_env python
conda activate segment_env
pip install -e .
pip install -e ".[demo]"

```
# Download model
```
cd checkpoints
bash download_ckpts.sh 
```

# Run AI-API
+ In segment_env environment
`python app.py`


# Save json coco format
+ File name: instances_test.json , requests `instances_` in file name



