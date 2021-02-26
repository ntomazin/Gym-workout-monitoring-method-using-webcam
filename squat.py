from bodyParts import *
import cv2

knee_deviation = 20 #pixels
squat_deviation = 40 # pixels
shoulder_deviation = 20 #pixels

def squat_counter(key_points: dict, got_down: bool):
    knees = getKnees()
    knee1Points = [key_points[x] for x in key_points.keys() if
                   str(knees[0]) in str(x)][0]
    knee2Points = [key_points[x] for x in key_points.keys() if
                   str(knees[1]) in str(x)][0]
    hips = getHips()
    hip1Points = [key_points[x] for x in key_points.keys() if
                  str(hips[0]) in str(x)][0]
    hip2Points = [key_points[x] for x in key_points.keys() if
                   str(hips[1]) in str(x)][0]

    if not got_down:
        #we checking if he is going for the squat
        # we will supose that a squat is done when hips get near the knees and when he gets back
        if knee1Points[1] - hip1Points[1] < squat_deviation and knee2Points[1] - hip2Points[1] < squat_deviation:
            return True
        return False
    else:
        pass



def checkKneeDistance(key_points: dict, frame):
    knees = getKnees()

    knee1Points = [key_points[x] for x in key_points.keys() if
                   str(knees[0]) in str(x)][0]
    knee2Points = [key_points[x] for x in key_points.keys() if
                   str(knees[1]) in str(x)][0]

    # check if knees are too close
    # we gonna compare them to the feet, they have to be wider than the feet
    feet = getFeet()

    feet1Points = [key_points[x] for x in key_points.keys() if
                   str(feet[0]) in str(x)][0]
    feet2Points = [key_points[x] for x in key_points.keys() if
                   str(feet[1]) in str(x)][0]

    if knee1Points[0] - feet1Points[0] > knee_deviation or knee2Points[0] - feet2Points[0] > knee_deviation:
        cv2.putText(frame, "Knees: BAD", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(frame, knee1Points, knee2Points, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        return False
    else:
        cv2.putText(frame, "Knees: OK", (10, 300), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        return True


def checkFeetDistance(key_points: dict, frame):
    shoulders = getShoulders()

    shoulder1Points = [key_points[x] for x in key_points.keys() if
                   str(shoulders[0]) in str(x)][0]
    shoulder2Points = [key_points[x] for x in key_points.keys() if
                   str(shoulders[1]) in str(x)][0]

    # check if feet are too close
    # we gonna compare them to the shoulders, they have to be wider than the shoulders
    feet = getFeet()

    feet1Points = [key_points[x] for x in key_points.keys() if
                   str(feet[0]) in str(x)][0]
    feet2Points = [key_points[x] for x in key_points.keys() if
                   str(feet[1]) in str(x)][0]

    if shoulder1Points[0] - feet1Points[0] < -shoulder_deviation or feet2Points[0] - shoulder2Points[0] < -shoulder_deviation:
        cv2.putText(frame, "Feet: BAD", (10, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(frame, feet1Points, feet2Points, (0, 0, 255), 3, lineType=cv2.LINE_AA)
        return False
    else:
        cv2.putText(frame, "Feet: OK", (10, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        return True



