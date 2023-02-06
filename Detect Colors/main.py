import cv2
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('Parameters')
cv2.createTrackbar('H_min', 'Parameters', 0, 179, nothing)
cv2.createTrackbar('H_max', 'Parameters', 0, 179, nothing)
cv2.createTrackbar('S_min', 'Parameters', 0, 255, nothing)
cv2.createTrackbar('S_max', 'Parameters', 0, 255, nothing)
cv2.createTrackbar('V_min', 'Parameters', 0, 255, nothing)
cv2.createTrackbar('V_max', 'Parameters', 0, 255, nothing)
cv2.createTrackbar('Kernel_X', 'Parameters', 1, 30, nothing)
cv2.createTrackbar('Kernel_Y', 'Parameters', 1, 30, nothing)

cv2.setTrackbarPos('H_min', 'Parameters', 10)     # green: 30   orange: 10
cv2.setTrackbarPos('H_max', 'Parameters', 32)     # green: 90   orange: 32
cv2.setTrackbarPos('S_min', 'Parameters', 135)    # green: 50   orange: 135
cv2.setTrackbarPos('S_max', 'Parameters', 255)
cv2.setTrackbarPos('V_min', 'Parameters', 135)    # green: 50   orange: 135
cv2.setTrackbarPos('V_max', 'Parameters', 255)
cv2.setTrackbarPos('Kernel_X', 'Parameters', 5)
cv2.setTrackbarPos('Kernel_Y', 'Parameters', 5)

video_path = './fruits.mp4'

cap = cv2.VideoCapture(video_path)
frame_counter = 0

font = cv2.FONT_HERSHEY_PLAIN
fontScale = 1
color = (0, 255, 0)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_counter += 1
    if frame_counter == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        frame_counter = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if ret:
        rsz = cv2.resize(frame, (400,300), fx=0, fy= 0, interpolation = cv2.INTER_CUBIC)
        hsv = cv2.cvtColor(rsz, cv2.COLOR_BGR2HSV)

        H_min = cv2.getTrackbarPos('H_min', 'Parameters')
        H_max = cv2.getTrackbarPos('H_max', 'Parameters')
        S_min = cv2.getTrackbarPos('S_min', 'Parameters')
        S_max = cv2.getTrackbarPos('S_max', 'Parameters')
        V_min = cv2.getTrackbarPos('V_min', 'Parameters')
        V_max = cv2.getTrackbarPos('V_max', 'Parameters')
        Kernel_X = cv2.getTrackbarPos('Kernel_X', 'Parameters')
        Kernel_Y = cv2.getTrackbarPos('Kernel_Y', 'Parameters')

        dark_color = np.array([H_min, S_min, V_min])
        bght_color = np.array([H_max, S_max, V_max])

        mask = cv2.inRange(hsv, dark_color, bght_color)

        Kernel = np.ones((Kernel_X, Kernel_Y), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, Kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernel)

        cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(rsz, cnts, -1, (0, 255, 255), 1)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 180:
                cv2.rectangle(rsz, (x, y), (x + w, y + h), (0, 255, 0), 1)

        cv2.imshow('Video', rsz)
        cv2.imshow('Mask', mask)

        k = cv2.waitKey(5)
        if k == 27:
            cv2.destroyAllWindows()
            break
cap.release()

