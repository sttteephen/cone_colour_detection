import torch
import cv2
import numpy as np

cap = cv2.VideoCapture("video.mp4")

model = torch.hub.load("ultralytics/yolov5", "custom", "best.pt")
print(model.conf, model.iou, model.agnostic, model.multi_label)  # NMS settings
model.agnostic = True

print(dir(model))

while True:
    # get video frame by frame
    ret, frame = cap.read()

    # do detection and get pandas data frame of results
    results = model(frame)
    resultspd = results.pandas().xyxy[0]

    # print(resultspd)
    # resultspd columns: [0] xmin, [1] ymin, [2] xmax, [3] ymax, [4] confidence, [5] class, [6] name

    # loop over results
    for i in range(len(resultspd.index)):
        # choose colour
        if int(resultspd.loc[i][5]) == 0:
            colour = (0, 255, 255)  # yellow
        elif int(resultspd.loc[i][5]) == 2:
            colour = (255, 0, 255)
        elif int(resultspd.loc[i][5]) == 3:
            colour = (255, 255, 0)  # blue

        # print(resultspd.loc[i][4])
        frame = cv2.rectangle(
            frame,
            (round(resultspd.loc[i][0]), round(resultspd.loc[i][1])),
            (round(resultspd.loc[i][2]), round(resultspd.loc[i][3])),
            colour,
            2,
        )

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
