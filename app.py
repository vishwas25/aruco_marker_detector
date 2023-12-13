import streamlit as st
import cv2 as cv2
import cv2.aruco as aruco
import numpy as np
import os as os

def loadAugImages(path):
    """
    :param path : folder in which all the marker images with ids are stored
    : return : dictionary with key as the id and values

    """

    myList = os.listdir(path)
    noOfMarkers = len(myList)
    print("Total number of Markers Detected: ",noOfMarkers)
    augDics = {}
    for imgPath in myList:
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        augDics[key] = imgAug
    return augDics

def findArucoMarkers(img,markerSize=6, totalMarkers=250, draw=True):
    """
    :param image in which to find the aruco makers
    :param markersize : the size of the markers
    :param totalMarkers : total number of markers that compose the dictionary
    :param draw : flag to draw bbox around markers detected
    : return : bounding boxexs and id numbers of markers detected

    """
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.getPredefinedDictionary(key)
    arucoParam = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(arucoDict, arucoParam)
    bboxs,ids, rejected = detector.detectMarkers(imgGray)

    print(ids)

    if draw:
        aruco.drawDetectedMarkers(img,bboxs)

    return [bboxs, ids]



def augmentAruco(bbox,id, img,imgAug, drawId=True):
    """
    :param bbox : four corner points of box
    :param id : marker id of the corresponding box used only for display
    :param img : final image on which to draw
    :param the image that will be overlapped on the marker
    :param drawId : flag t display id of the marker
    : return : image with the augment image

    """
    tl = (int(bbox[0][0][0]), int(bbox[0][0][1]) )
    tr = bbox[0][1][0], bbox[0][1][1]
    br = bbox[0][2][0], bbox[0][2][1]
    bl = bbox[0][3][0], bbox[0][3][1]


    h,w,c = imgAug.shape

    pts1 = np.array([tl,tr,br,bl])
    pts2 = np.float32([[0,0],[w,0],[w,h],[0,h]])

    matrix, _ = cv2.findHomography(pts2,pts1)
    imgOut = cv2.warpPerspective(imgAug,matrix,(img.shape[1],img.shape[0]))
    cv2.fillConvexPoly(img,pts1.astype(int),(0,0,0))
    imgOut = img + imgOut

    if drawId:
        cv2.putText(imgOut,str(id), tl, cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

    return imgOut


def main():
    st.title("Aruco Marker Detection")

    # OpenCV video capture
    cap = cv2.VideoCapture(-1)
    
    # Wait for the camera to open successfully
    while not cap.isOpened():
        st.warning("Error: Unable to open the camera. Please check if the camera is connected.")
        st.stop()

    augDics = loadAugImages("Markers")

    # Streamlit placeholder for displaying the camera feed
    image_placeholder = st.image([], channels="BGR", use_column_width=True)

    while True:
        success, img = cap.read()

        # Check if the camera read was successful
        if not success:
            st.warning("Error: Unable to read from the camera.")
            break

        arucoFound = findArucoMarkers(img)

        # Loop through all the markers and augment each one
        if len(arucoFound[0]) != 0:
            for bbox, marker_id in zip(arucoFound[0], arucoFound[1]):
                if int(marker_id) in augDics.keys():
                    img = augmentAruco(bbox, marker_id, img, augDics[int(marker_id)])

        # Display the camera feed in the Streamlit app
        image_placeholder.image(img, channels="BGR")

        # Wait for a key event and break the loop if the 'Esc' key is pressed
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Release the camera and close the OpenCV window when the app is closed
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
