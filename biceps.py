from bodyParts import *
import cv2

shoulder_deviation = 20 #pixels
elbow_deviation = 20 #pixels

def right_biceps_counter(key_points: dict, got_up: bool):
    shoulders = getShoulders()
    shoulder2Points = [key_points[x] for x in key_points.keys() if
                   str(shoulders[1]) in str(x)][0]
    hands = getHands()
    hand2Points = [key_points[x] for x in key_points.keys() if
                   str(hands[1]) in str(x)][0]

    if not got_up:
        # we will supose that a biceps repetition is done when hand gets near the shoulder and when he gets back
        if hand2Points[1] - shoulder2Points[1] < shoulder_deviation:
            return True
        return False

def left_biceps_counter(key_points: dict, got_up: bool):
    shoulders = getShoulders()
    shoulder1Points = [key_points[x] for x in key_points.keys() if
                       str(shoulders[0]) in str(x)][0]
    hands = getHands()
    hand1Points = [key_points[x] for x in key_points.keys() if
                   str(hands[0]) in str(x)][0]

    if not got_up:
        # we will supose that a biceps repetition is done when hand gets near the shoulder and when he gets back
        if hand1Points[1] - shoulder1Points[1] < shoulder_deviation:
            return True
        return False

def check_if_exercise_done_right(key_points: dict, got_up:bool):
    elbows = getElbow()
    elbow2Points = [key_points[x] for x in key_points.keys() if
                   str(elbows[1]) in str(x)][0]

    hands = getHands()
    hand2Points = [key_points[x] for x in key_points.keys() if
                   str(hands[1]) in str(x)][0]

    if got_up:
        if hand2Points[1] - elbow2Points[1] > elbow_deviation:
            return True
        return False

def check_if_exercise_done_left(key_points: dict, got_up:bool):
    elbows = getElbow()
    elbow1Points = [key_points[x] for x in key_points.keys() if
                   str(elbows[0]) in str(x)][0]
    hands = getHands()
    hand1Points = [key_points[x] for x in key_points.keys() if
                   str(hands[0]) in str(x)][0]
    if got_up:
        if hand1Points[1] - elbow1Points[1] > elbow_deviation:
            return True
        return False

