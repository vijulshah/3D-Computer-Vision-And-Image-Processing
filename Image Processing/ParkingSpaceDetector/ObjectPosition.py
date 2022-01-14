import cv2
import pickle

width, height = 107, 48

try:
    with open('ObjectPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

def onMouseClick(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1+width and y1 < y < y1+height:
                posList.pop(i)

    with open('ObjectPos', 'wb') as f:
        pickle.dump(posList, f)

while True:
    img = cv2.imread('carParkImg.png')
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0]+width, pos[1]+height), (0, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", onMouseClick)
    cv2.waitKey(1)
