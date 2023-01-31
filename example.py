print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cvzone
import numpy as np
import cv2
import utils
import solver
import random

width,height = 450,450
model = utils.initializeModel()
lst = os.listdir('sudokus')
path = os.path.join('sudokus',random.choice(lst))


img = cv2.imread(path)
img = cv2.resize(img,(width,height))
imgBlank = np.zeros((height,width,3),np.uint8)
imgThresh = utils.preprocess(img)
imgContours = img.copy()
imgBigContours = img.copy()
contours,_ = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgContours,contours,-1,(0,0,255),10)
biggest,maxArea = utils.biggestContours(contours)

if biggest.size != 0:
    cv2.drawContours(imgBigContours,biggest,-1,(0,0,255),10)
    biggest = utils.reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imageWrapedColoured = cv2.warpPerspective(img,matrix,(width,height))
    imageWrapedColoured = cv2.cvtColor(imageWrapedColoured,cv2.COLOR_BGR2GRAY)
    imageDetectedDigits = imgBlank.copy()

    imgSolvedDigits = imgBlank.copy()
    boxes = utils.splitBoxes(imageWrapedColoured)

    print("Almost there..")

    numbers = utils.getPredictions(boxes,model)
    imageDetectedDigits = utils.displayNumbers(imageDetectedDigits,numbers,(0,255,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0,0,1)
    board = np.array_split(numbers,9)
    try:
        solver.solve(board)
    except:
        pass
    flatlist = []

    for sublist in board:
        for item in sublist:
            flatlist.append(item)
    solvedNumbers = flatlist*posArray
    imgSolvedDigits = utils.displayNumbers(imgSolvedDigits,solvedNumbers)
    
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits,matrix,(width,height))
    inv_prespective = cv2.addWeighted(imgInvWarpColored,1,img,0.5,1)


imgArray = ([img,imgThresh,imgContours,imgBigContours,imageWrapedColoured,imageDetectedDigits,imgSolvedDigits,inv_prespective])
stackedImage = cvzone.stackImages(imgArray,4,0.7)
print('Done.')
cv2.imshow('win',stackedImage)
cv2.waitKey(0)
