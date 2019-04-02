import numpy as np
import cv2
from PIL import Image, ImageEnhance
#from matplotlib import pyplot as plt

def func(frame):
    # image = Image.open(frame)
    # contrast = ImageEnhance.Contrast(image)
    # img=contrast.enhance(2)
    # img = np.asarray(img)
    # r, g, b = cv2.split(img)
    # contrast=cv2.merge([b, g, r])
    # # Reads the enhanced image and converts it to grayscale, creates new file
    # converted2 = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    hand_cascade = cv2.CascadeClassifier('hand.xml')

    # Reads frame, not path.
    #frame = cv2.resize(frame,(128,128))
    #converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #cv2.imshow("original",converted2)

    hands = hand_cascade.detectMultiScale(frame,1.1,5)
    print(hands)

    for(x,y,w,h) in hands:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)
        faceimg = frame[ny:ny+nr, nx:nx+nr]

    #cv2.imshow("cropped",faceimg)
    '''
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    #frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    #skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    #skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin,60,60)
    #cv2.imshow("edge detection",img2)
     
    hog = cv2.HOGDescriptor()
    h = hog.compute(img2)
    print(len(h))
    
    surf = cv2.xfeatures2d.SURF_create()
    #surf.extended=True
    img2 = cv2.resize(img2,(256,256))
    kp, des = surf.detectAndCompute(img2,None)
    #print(len(des))
    img2 = cv2.drawKeypoints(img2,kp,None,(0,0,255),4)
    #plt.imshow(img2),plt.show()
    '''
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(len(des))
    #return des

def func2(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(128,128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #cv2.imshow("original",converted2)

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    
    #cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin,60,60)
    #cv2.imshow("edge detection",img2)
    img2 = cv2.resize(img2,(256,256))
    orb = cv2.xfeatures2d.ORB_create()
    kp, des = orb.detectAndCompute(img2,None)

    #print(len(des2))
    img2 = cv2.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
    #plt.imshow(img2),plt.show()
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return des2

img = cv2.imread("001.jpg")
#print(img)
cv2.imshow('image',img)
func(img)
