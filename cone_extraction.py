import os
import json
import cv2
import numpy as np

FSOCO_FOLDER = "/Users/stephenmesser/Downloads/fsoco_bounding_boxes_train"
OUTPUT = "/Users/stephenmesser/Desktop/extracted_cones"

COULOR_DICT = {
    "blue_cone": 0,
    "yellow_cone": 1,
    "orange_cone": 2,
    "large_orange_cone": 3,
    "unknown_cone": 4,
}

for team_folder in os.listdir(os.fsdecode(FSOCO_FOLDER)):
    if os.path.isdir(FSOCO_FOLDER + "/" + team_folder):
        anns_folder = f"{FSOCO_FOLDER}/{team_folder}/ann"
        imgs_folder = f"{FSOCO_FOLDER}/{team_folder}/img"

        for img_name in os.listdir(imgs_folder):
            ann = f"{anns_folder}/{img_name}.json"
            f = open(ann)

            img = cv2.imread(f"{imgs_folder}/{img_name}")
            boxes = json.load(f)

            for box in boxes["objects"]:
                colour = box["classTitle"]

                if colour in COULOR_DICT.keys():
                    point1 = box["points"]["exterior"][0]
                    point2 = box["points"]["exterior"][1]

                    x1 = int(point1[0])
                    x2 = int(point2[0])
                    y1 = int(point1[1])
                    y2 = int(point2[1])
                    # print(x1, x2, y1, y2)

                    cone = img[y1:y2, x1:x2]
                    # print(cone)

                    out_name = f"{OUTPUT}/{colour}_{os.fsdecode(img_name)}"
                    cv2.imwrite(out_name, cone)
