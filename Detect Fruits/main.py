import cv2
import numpy as np

font = cv2.FONT_HERSHEY_PLAIN
color = (0, 255, 0)

# Fruit names
fname = ['Orange', 'Apple']

video_path = './fruits.mp4'

cap = cv2.VideoCapture(video_path)
frame_counter = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_counter += 1
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if ret:
        rsz = cv2.resize(frame, (400,300), fx=0, fy= 0, interpolation = cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)

        min = []
        max = []
        cnts = []
        mask = []

        # Orange HSV color range
        min.append(np.array([10, 135, 135], np.uint8))
        max.append(np.array([32, 255, 255], np.uint8))

        # Green HSV color range
        min.append(np.array([30, 50, 50], np.uint8))
        max.append(np.array([90, 255, 255], np.uint8))

        # Kernel filter
        Kernel = np.ones((5, 5), np.uint8)

        for i in range(2):
            mask.append(cv2.inRange(hsv, min[i], max[i]))

            # Morphological transformations with orange mask and kernel
            mask[i] = cv2.morphologyEx(mask[i], cv2.MORPH_CLOSE, Kernel)
            mask[i] = cv2.morphologyEx(mask[i], cv2.MORPH_OPEN, Kernel)

            # Finding contours
            cnts.append(cv2.findContours(mask[i], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0])

            cv2.drawContours(rsz, cnts[i], -1, (0, 255, 255), 1)

            for c in cnts[i]:
                x, y, w, h = cv2.boundingRect(c)
                if w > 180 and w < 230 and h < 230:
                    cv2.rectangle(rsz, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(rsz, fname[i], (x, y - 10), font, 1.5, color, 1)

        cv2.imshow('Video', rsz)
        cv2.imshow('HSV', hsv)
        cv2.imshow('Orange Mask', mask[0])
        cv2.imshow('Apple Mask', mask[1])

        k = cv2.waitKey(5)
        if k == 27:
            cv2.destroyAllWindows()
            break

cap.release()

