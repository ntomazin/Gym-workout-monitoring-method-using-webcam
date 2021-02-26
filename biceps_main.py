import time
import numpy as np
from biceps import *
from utils import *
import sys
#TODO namjestiti ROI
#TODO napraviti config file sa pathovima i ROIovima

device = "gpu"
num_of_frames = 20
video = "videos/biceps_lijeva.mp4" #"videos/biceps_lijeva.mp4" # "videos/biceps_dvorucno.mp4" # "videos/biceps_desna.mp4"
MODE = "MPI"
ROI_coordinates = ((150, 25), (500, 400)) #namjesiti ROI
got_down_left = False
got_down_right = False


if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12],
                  [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]

elif MODE is "MPI":
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

inWidth = 368
inHeight = 368
threshold = 0.1

cap = cv2.VideoCapture(video)

hasFrame, frame = cap.read()
video_output = video.replace(".mp4", "_output.avi")
vid_writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), num_of_frames,
                             (frame.shape[1], frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

key_points = dict()
frameCounter = 0
bicepsCounterLeft = 0
bicepsCounterRight = 0
ROIActive = True
initial_position = dict()

while cv2.waitKey(1) < 0:
    frameCounter += 1
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))

        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:

            if (partA, frameCounter) not in key_points.keys():
                key_points[(partA)] = points[partA]
            if (partB, frameCounter) not in key_points.keys():
                key_points[(partB)] = points[partB]

            cv2.line(frame, points[partA], points[partB], (0, 0, 255), 3, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(partA), points[partA], cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, str(partB), points[partB], cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Reps right: " + str(bicepsCounterLeft), (10, 270), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Reps left: " + str(bicepsCounterRight), (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (10, 20), cv2.FONT_HERSHEY_COMPLEX, .8,
                (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    ROIActive = is_in_ROI(key_points, ROI_coordinates, [2, 3, 4, 5, 6, 7])
    if ROIActive:
        # if person is in ROI
        got_down_new_left = left_biceps_counter(key_points, got_down_left)
        got_down_new_right = right_biceps_counter(key_points, got_down_right)
        if len(initial_position) == 0:
            initial_position = key_points.copy()

        if got_down_right != got_down_new_right and got_down_new_right == True:
            bicepsCounterRight += 1
            got_down_right = got_down_new_right
        if got_down_left != got_down_new_left and got_down_new_left == True:
            bicepsCounterLeft += 1
            got_down_left = got_down_new_left

        if got_down_left == True and check_if_exercise_done_left(key_points, got_down_left):
            #when it falls bellow elbows
            got_down_left = False
        if got_down_right == True and check_if_exercise_done_right(key_points, got_down_right):
            #when it falls bellow elbows
            got_down_right = False
    else:
        # else put the ROI and write the text
        image = cv2.rectangle(frame, ROI_coordinates[0], ROI_coordinates[1], (0, 0, 255), 2)
        cv2.putText(frame, "Not in ROI", (250, 470), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Output-Skeleton', frame)
    vid_writer.write(frame)

print(sys.getsizeof(key_points))
vid_writer.release()
