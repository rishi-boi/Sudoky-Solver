import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initializing model
def initializeModel():
    model = load_model('models/model-OCR.h5')
    return model

# Preprocessing image
def preprocess(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,1,1,15,2)
    return imgThreshold

# Reordering points
def reorder(points):
    points = points.reshape((4,2))
    pointsNew = np.zeros((4,1,2),dtype=np.int32)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis = 1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]

    return pointsNew

# Spliting sudoku into 81 cells
def splitBoxes(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    
    return boxes

# Predicting numbers in each cell of sudoku
def getPredictions(boxes,model):
    result = []
    for image in boxes:
        # image = image
        # w,h,c = image.shape
        # image = image[5:h - 5, 5:w - 5]
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
        img = cv2.resize(img,(48,48))
        img = img / 255
        img = img.reshape(1,48,48,1)
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis = 1)
        probabilityValue = np.amax(predictions)
        if probabilityValue > 0.90:
            result.append(classIndex[0])
        else:
            result.append(0)
        
    return result

# Displaying number on blank screen
def displayNumbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1] / 9)
    secH = int(img.shape[0] / 9)
    for x in range(0,9):
        for y in range(0,9):
            if numbers[(y*9) + x] != 0:
                cv2.putText(img,str(numbers[(y*9) + x]),(x*secW+int(secW/2) - 10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,1.5,color,2,cv2.LINE_AA)
    
    return img

# Finding biggest contours
def biggestContours(contours):
    biggest= np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    
    return biggest,max_area