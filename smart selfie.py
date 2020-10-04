import cv2 as cv
import numpy as np
import dlib

#Initialize dlib's face detector (HOG-based) and then create the facial landmark detector
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor.dat")

#indices for the coordinates corresponding to the mouth
(mStart,mEnd)=(48,67)

smile_const=5
counter=0
selfie_no=0

#Function to convert the rectangle predicted by dlib
#to bounding box in OpenCV with the format (x,y,w,h)
def rect_to_bb(rect):
    x=rect.left()
    y=rect.top()
    w=rect.right()-x
    h=rect.bottom()-y

    #return a tuple of (x,y,w,h)
    return (x,y,w,h)

#function to convert the facial coordinates recognized by the predictor
#into a numpy array to ease further usage of the points
def shape_to_np(shape,dtype="int"):
    #initialize the list of (x,y) coordinates
    coords=np.zeros((68,2),dtype=dtype)

    #Loop over the 68 facial landmarks and convert them to a 2-tuple of (x,y) coordinates
    for i in range(0,68):
        coords[i]=(shape.part(i).x,shape.part(i).y)

    #Return the list of (x,y) coordinates
    return coords

def smile(shape):
    left=shape[48]
    right=shape[54]

    #Average of the pints in the center of the mouth
    mid=(shape[51]+shape[62]+shape[66]+shape[57])/4
    
    #perpendicular distance between the mid and the line joining left and right
    dist=np.abs(np.cross(right-left,left-mid))/np.linalg.norm(right-left)
    return dist
    
cam=cv.VideoCapture(0)

while(cam.isOpened()):
    

    ret,image=cam.read()
    image=cv.flip(image,1)
    gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    #Detect faces in the grayscale image
    rects=detector(gray,2)

    #Loop over each face, to draw a rectangle around it
    for i in range(0,len(rects)):

        #Convert dlib's rectangle to an OpenCv-style bouding box
        #i.e, (x,y,w,h), then draw the face bounding box
        (x,y,w,h)=rect_to_bb(rects[i])
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

        #show the face number
        cv.putText(image,"Face #()".format(i+1),(x-10,y-10),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        #Determine the facial landmarks for the face region
        shape=predictor(gray,rects[i])

        #Convert the facial landmark (x,y) coordinates into a numpy array
        shape=shape_to_np(shape)

        mouth=shape[mStart:]
        
        #Loop over the (x,y) coordinates for the facial landmarks and draw them on the image
        for (x,y) in mouth:
            cv.circle(image,(x,y),1,(255,255,255),-1)

        #smile parameter from the mouth
        smile_param=smile(shape)
        cv.putText(image,"SP:(:.2f)".format(smile_param),(300,30),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        if smile_param>smile_const:
            cv.putText(image,"Smile detected".format(smile_param),(300,60),cv.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            counter+=1
            if counter>=15: #if smile is sustained for 15 frames,take selfie
                selfie_no+=1
                ret,frame=cam.read()
                img_name="smart_selfie_{}.png".format(selfie_no)
                cv.imwrite(img_name,frame)
                print("{} taken!".format(imag_name))
                counter=0 #reset counter once selfie is taken

        else:
            counter=0 #reset counter once smile is not detected in a frame
            

            
        cv.imshow('live_face',image)
        key=cv.waitKey(1)
        if key==27:
            break
        
        


cam.release()
cv.destroyAllWindows()

