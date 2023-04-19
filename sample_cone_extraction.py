import os
import json
import cv2
import numpy as np

IMAGES = "/Users/stephenmesser/Downloads/fsoco_sample/images"
ANNOTATIONS = "/Users/stephenmesser/Downloads/fsoco_sample/bounding_boxes"
OUTPUT = "/Users/stephenmesser/Downloads/fsoco_sample/extracted_cones"


directory = os.fsdecode(IMAGES)

for file in os.listdir(directory):

    read_name = os.fsdecode(file)
    boxes_path = f"{ANNOTATIONS}/{read_name}.json"
    image_path = f"{IMAGES}/{read_name}"

    img = cv2.imread(image_path)
    f = open(boxes_path)

    boxes = json.load(f)
    for box in boxes["objects"]:

        colour = box["classTitle"]
        point1 = box["points"]["exterior"][0]
        point2 = box["points"]["exterior"][1]

        x1 = int(point1[0])
        x2 = int(point2[0])
        y1 = int(point1[1])
        y2 = int(point2[1])

        cone = cone = img[y1:y2, x1:x2]

        out_name = f"{OUTPUT}/{colour}_{os.fsdecode(file)}"
        cv2.imwrite(out_name, cone)
