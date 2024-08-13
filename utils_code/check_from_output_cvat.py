import json

# Specify the path to the JSON file
file_path = '/home/labelling/Project/segment-anything-2/data/check_data/annotations/instances_default.json'

# Open the JSON file
with open(file_path) as json_file:
    data = json.load(json_file)

# Now you can work with the data from the JSON file
# For example, you can access a specific key like this:
count = data['annotations'][0]['segmentation']['counts']
print(data['annotations'])