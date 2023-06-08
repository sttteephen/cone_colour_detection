import torch
import cv2
import numpy as np
from tensorflow import keras
import time
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)

yolo_model = torch.hub.load("ultralytics/yolov5", "custom", "best.pt")
yolo_model.agnostic = True
# print(model.conf, model.iou, model.agnostic, model.multi_label)  # NMS settings

colour_model = keras.models.load_model(
    "/Users/stephenmesser/Desktop/FS-AI/cone_colour_detection/cones_modelreal_small.keras"
)

while True:
    start = time.time()

    # get video frame by frame
    ret, frame = cap.read()
    frame = frame[
        int(frame.shape[0] / 2) : int(frame.shape[0]) - 100,
        75 : int(frame.shape[1]) - 75,
    ]

    # do yolo detection and get pandas data frame of results
    results = yolo_model(frame)
    resultspd = results.pandas().xyxy[0]

    # resultspd columns: [0] xmin, [1] ymin, [2] xmax, [3] ymax, [4] confidence, [5] class, [6] name

    # loop over results
    for i in range(len(resultspd.index)):
        x1 = round(resultspd.loc[i][0])
        y1 = round(resultspd.loc[i][1])
        x2 = round(resultspd.loc[i][2])
        y2 = round(resultspd.loc[i][3])

        cone = frame[y1:y2, x1:x2]
        if cone.shape[0] > 10:
            cone = cv2.resize(cone, dsize=(25, 25))
            cone_arr = [cone]

            # get cone colour from nn
            colour_prediction = colour_model.predict(np.asarray(cone_arr)).argmax()
            # colour_prediction = int(resultspd.loc[i][5])
            colour = (0, 0, 0)
            if colour_prediction == 0:
                colour = (255, 190, 0)  # blue
            elif colour_prediction == 1:
                colour = (0, 255, 255)  # yellow
            elif colour_prediction == 2:
                colour = (0, 117, 255)  # orage
            elif colour_prediction == 3:
                colour = (0, 0, 255)  # large orange

            frame = cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                colour,
                1,
            )

    cv2.imshow("frame", frame)

    end = time.time()
    print(end - start)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
