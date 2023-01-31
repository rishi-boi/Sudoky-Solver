print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cvzone
import numpy as np
import cv2
import utils
import solver
import argparse
import datetime

# saving file name
filename = 'SolvedImage(' + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.') + ').jpg'

# Defining height and width of image
width,height = 450,450

# Initializing model
model = utils.initializeModel()

parser = argparse.ArgumentParser()
parser.add_argument('path',help="Enter path of image. Note:- it should not contain any spaces between letters example:-'Sudo ku'")

args = parser.parse_args()
path = args.path
# saving dir name
dirName = os.path.dirname(path)

filename = os.path.join(dirName,filename)

# Reading Image
img = cv2.imread(path)

# Resizing image
img = cv2.resize(img,(width,height))

# Creating blank images
imgBlank = np.zeros((height,width,3),np.uint8)

# Processing image
imgThresh = utils.preprocess(img)
imgContours = img.copy()
imgBigContours = img.copy()

# Finding contours
contours,_ = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Drawing contours
cv2.drawContours(imgContours,contours,-1,(0,0,255),10)

# Finding biggest contour in image
biggest,maxArea = utils.biggestContours(contours)

if biggest.size != 0:
    cv2.drawContours(imgBigContours,biggest,-1,(0,0,255),10)

    # Reordering contours points
    biggest = utils.reorder(biggest)

    # Wraping Image
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imageWrapedColoured = cv2.warpPerspective(img,matrix,(width,height))
    imageWrapedColoured = cv2.cvtColor(imageWrapedColoured,cv2.COLOR_BGR2GRAY)
    imageDetectedDigits = imgBlank.copy()

    imgSolvedDigits = imgBlank.copy()

    # Getting each cell of sudoku
    boxes = utils.splitBoxes(imageWrapedColoured)
    
    print('almost there..')

    # Predicting number in each cell of sudoku
    numbers = utils.getPredictions(boxes,model)

    # Displaying numbers of sudoku in blank screen
    imageDetectedDigits = utils.displayNumbers(imageDetectedDigits,numbers,(0,255,255))
    numbers = np.asarray(numbers)
    posArray = np.where(numbers > 0,0,1)
    
    # Solving sudoku in backend
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

    # Displaying solved sudoku in black screen
    imgSolvedDigits = utils.displayNumbers(imgSolvedDigits,solvedNumbers)
    
    # Rewraping image
    pts2 = np.float32(biggest)
    pts1 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits,matrix,(width,height))
    inv_prespective = cv2.addWeighted(imgInvWarpColored,1,img,0.5,1)

# Stacking all images
imgArray = ([img,imgThresh,imgContours,imgBigContours,imageWrapedColoured,imageDetectedDigits,imgSolvedDigits,inv_prespective])
cv2.imwrite(filename,inv_prespective)
stackedImage = cvzone.stackImages(imgArray,4,0.7)
print('Done.')
cv2.imshow('win',stackedImage)
cv2.waitKey(0)