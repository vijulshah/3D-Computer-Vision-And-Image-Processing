import cv2
import numpy as np
# Download (trained model) cfg and weights file from: https://pjreddie.com/darknet/yolo/
# Here I have downloaded one tiny version (poor model) for faster computation and one strong model which will take a bit more time than tiny version. But its accuracy is high. Results can be seen by comparing input_img & output_img vs input_img_tiny & output_img_tiny.

# -----------------------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
do_process = False

while True:
    success, img = cap.read()
    if not success:
        break
    cv2.imshow("Feed", img)
    k = cv2.waitKey(1)
    if k % 256 == 27: # escape key
        break
    elif k % 256 == 32: # space key
        cv2.imwrite("input_img.png", img)
        do_process = True
        break
cap.release()

# -----------------------------------------------------------------------------------------
WHT = 320
CONFIDENCE_THRESHOLD = 0.50 # 50%
NMS_THRESHOLD = 0.3 # 30% # the more lower it is, more aggressive it will be.

classesFile = './labels' # total 80 labels
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip("\n").split('\n')

modelConfiguration = 'model.cfg'
modelWeights = 'model.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
print("layerNames:\n",layer_names)

outputnames = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print("outputNames:\n",outputnames)

# -----------------------------------------------------------------------------------------
def findObjects(outputs,img):
    ht, wt, ct = img.shape
    bounding_box = []
    class_ids = []
    confidence_values = []

    for op in outputs:
        for detection in op:
            # probability of classes starts after first 5 columns. See Explaination_1.png
            scores = detection[5:]
            # find which of these classes have highest probabilities.
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Now, filter our objects.
            if confidence > CONFIDENCE_THRESHOLD: # then only it is a good detection
                w, h = int(detection[2] * wt), int(detection[3] * ht) # as detection[2] = width & detection[3] = height in percentage, we will multipy with the image's real height & width
                x, y = int(detection[0] * wt - w/2), int(detection[1] * ht - h/2) # these are the center points. Div widht & height of the predicted object by 2 and minus from the real image
                bounding_box.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    print("len of bounding_box = ",len(bounding_box))
    # Sometimes, many boxes overlap each other if they detect the same object. So we have to remove one these boxes.
    # Function = Non-maximum supression. Eleminates overlapping boxes. Based on overlapping boxes, it will pick up box with max probability and suppress all the other boxes.
    indices = cv2.dnn.NMSBoxes(bounding_box, confidence_values, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    for i in indices:
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        print("Detected Object ==== ",classNames[class_ids[i]].upper())
        cv2.imwrite("output_img.png",cv2.putText(img,f'{classNames[class_ids[i]].upper()} {int(confidence_values[i]*100)}%', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2))

# -----------------------------------------------------------------------------------------
if do_process:
    img = cv2.imread("input_img.png")
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    blob = cv2.dnn.blobFromImage(img, 1/225, (WHT, WHT), (0,0,0), True, crop=False)

    for b in blob:
        for n, img_blob in enumerate(b):
            cv2.imshow(str(n), img_blob)

    net.setInput(blob)
    outputs = net.forward(outputnames)
    print(len(outputs))
    print(type(outputs))
    print(outputs[0].shape) 
    # (300, 85) 
    # 300 - bounding boxes, 
    # for rest 85:
    # one center-x, one center-y, one width, one height, one confidence - object present or not in bonding box. 
    # Rest 80 are probability of predictions of each class(labels)
    # E.g: if label/class 3 = 0.9 then we have 90% chance that the object is car. (For Rest of the labels have probability = 0)
    # See the Explaination_1.png
    print(outputs[1].shape) # (1200, 85)
    print("First Row:\n",outputs[0][0])

    cv2.imshow("Image", img)
    findObjects(outputs,img)

cv2.destroyAllWindows()
