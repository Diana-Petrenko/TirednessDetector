from argparse import ArgumentParser
from msvcrt import kbhit
from tracemalloc import stop
import numpy as np
import cv2
import dlib
import threading
from playsound import playsound
from collections import deque


global remind
global stop_threads
global k
remind = False
stop_threads = False


def get_left_eye_height(face_landmarks: np.ndarray):
    left_eye_top_x = (face_landmarks[37, 0] + face_landmarks[38, 0]) / 2.
    left_eye_top_y = (face_landmarks[37, 1] + face_landmarks[38, 1]) / 2.
    left_eye_bot_x = (face_landmarks[40, 0] + face_landmarks[41, 0]) / 2.
    left_eye_bot_y = (face_landmarks[40, 1] + face_landmarks[41, 1]) / 2.
    return np.sqrt(
        (left_eye_top_x - left_eye_bot_x)**2 + \
        (left_eye_top_y - left_eye_bot_y)**2)


def get_right_eye_height(face_landmarks):
    right_eye_top_x = (face_landmarks[43, 0] + face_landmarks[44, 0]) / 2.
    right_eye_top_y = (face_landmarks[43, 1] + face_landmarks[44, 1]) / 2.
    right_eye_bot_x = (face_landmarks[46, 0] + face_landmarks[47, 0]) / 2.
    right_eye_bot_y = (face_landmarks[46, 1] + face_landmarks[47, 1]) / 2.
    return np.sqrt(
        (right_eye_top_x - right_eye_bot_x)**2 + \
        (right_eye_top_y - right_eye_bot_y)**2)


def get_left_eye_width(face_landmarks):
    return np.sqrt(
        (face_landmarks[39, 0] - face_landmarks[36, 0])**2 + \
        (face_landmarks[39, 1] - face_landmarks[36, 1])**2)


def get_right_eye_width(face_landmarks):
    return np.sqrt(
        (face_landmarks[45, 0] - face_landmarks[42, 0])**2 + \
        (face_landmarks[45, 1] - face_landmarks[42, 1])**2)


def get_mouth_height(face_landmarks):
    return np.sqrt(
        (face_landmarks[57, 0] - face_landmarks[51, 0])**2 + \
        (face_landmarks[57, 1] - face_landmarks[51, 1])**2)


def get_mouth_width(face_landmarks):
    return np.sqrt(
        (face_landmarks[54, 0] - face_landmarks[48, 0])**2 + \
        (face_landmarks[54, 1] - face_landmarks[48, 1])**2)


def get_mouth_diag_1(face_landmarks):
    return np.sqrt(
        (face_landmarks[56, 0] - face_landmarks[50, 0])**2 + \
        (face_landmarks[56, 1] - face_landmarks[50, 1])**2)


def get_mouth_diag_2(face_landmarks):
    return np.sqrt(
        (face_landmarks[58, 0] - face_landmarks[52, 0])**2 + \
        (face_landmarks[58, 1] - face_landmarks[52, 1])**2)
def playmusic():
    global remind
    global stop_threads
    while not stop_threads:
        if remind:
            playsound(r'C:\Users\User\Desktop\Projects\Rise\voicealarm.mp3')
            remind=False

            


parser = ArgumentParser()
parser.add_argument('-v', '--write_video', type=str, default=None, required=False)
parser.add_argument('-l', '--logging', type=str, default='logging.txt', required=False)
args = parser.parse_args()

# if args.write_video is not None:
#     out_video_writer = cv2.VideoWriter(
#         args.write_video,
#         cv2.VideoWriter_fourcc(*'DIVX'),
#         10, (640, 480))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\User\Desktop\Projects\Rise\shape_predictor_68_face_landmarks.dat")
#due to the large size of the shape_predictor_68_face_landmarks-file, download it via the following link:https://drive.google.com/file/d/1_BwO9t3zREiE_mMSLiv8m-ZQ4Vjyikg_/view?usp=sharing

left_eye_closed = deque(maxlen=5)
right_eye_closed = deque(maxlen=5)
is_yawning = deque(maxlen=5)

with open(args.logging,"w") as f:
    cap = cv2.VideoCapture(0)
    counter = 0 
    musicthr=threading.Thread(target=playmusic).start()

    while True:
        ret, frame = cap.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(frame, 1)
        for rect in rects:
            face_landmarks = predictor(frame, rect)
            face_landmarks_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                face_landmarks_np[i] = (face_landmarks.part(i).x, face_landmarks.part(i).y)
                f.write(str(face_landmarks.part(i).x) + ', ' + str(face_landmarks.part(i).y) + ', ')
            f.write("\n")
            for i, (x, y) in enumerate(face_landmarks_np):
                cv2.circle(frame, (x, y), 1, (51, 153, 255), -1)
          
            # Detect drowsiness
            eye_limit=0.25
            mouth_limit=85

            left_eye_indicator = get_left_eye_height(face_landmarks_np)/get_left_eye_width(face_landmarks_np)
            if(left_eye_indicator<=eye_limit):
                left_eye_closed.append(1)
            else:
                left_eye_closed.append(0)
            
            right_eye_indicator = get_right_eye_height(face_landmarks_np)/get_right_eye_width(face_landmarks_np)
            if(right_eye_indicator<=eye_limit):
                right_eye_closed.append(1)
            else:
                right_eye_closed.append(0)

            # mouth_indicator = get_mouth_height(face_landmarks_np)/get_mouth_width(face_landmarks_np)
            mouth_indicator = get_mouth_diag_1(face_landmarks_np) + get_mouth_diag_2(face_landmarks_np)
            if(mouth_indicator>=mouth_limit):
                is_yawning.append(1)
            else:
                is_yawning.append(0)
  
            # Just make it visible
            bg = np.ones(shape=(frame.shape[0] + 6, frame.shape[1] + 6, 3), dtype=np.uint8)
            print(left_eye_closed, right_eye_closed, is_yawning)
            print(left_eye_indicator, right_eye_indicator, mouth_indicator)
            if (sum(list(left_eye_closed)) == len(left_eye_closed) \
                    and sum(list(right_eye_closed)) == len(right_eye_closed)) \
                    or sum(list(is_yawning)) == len(is_yawning):
                bg[...] = np.array([0, 0, 255], dtype=np.uint8)
                counter+=1
                if(counter==1):
                  remind=True
            else:
                   counter=0
                   remind = False
            bg[3:-3, 3:-3, :] = frame
            frame = bg
                
        cv2.imshow('videostream', frame)

        # if args.write_video is not None:
        #     out_video_writer.write(frame)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

cap.release()
# if args.write_video is not None:
#     out_video_writer.release()
cv2.destroyAllWindows()
stop_threads = True

