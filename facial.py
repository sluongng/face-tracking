import face_recognition
import cv2
import json

from os import listdir, getcwd
from os.path import isfile, join, splitext

_IS_VERBOSE = 0

if _IS_VERBOSE:
    print("Initiating facial.py !!!")
    print("Face_Recognition version is:", face_recognition.__version__)
    print("OpenCV version is:", cv2.__version__)

_WORKING_DIR = getcwd()
_IMAGES_DIR = join(_WORKING_DIR, "images")

course_facial_data = []

if _IS_VERBOSE:
    print("Working directory is:", _WORKING_DIR)
    print("Images directory is:", _IMAGES_DIR)


def _init_images():
    facial_encoding_data = []

    files = [f for f in listdir(_IMAGES_DIR) if isfile(join(_IMAGES_DIR, f))]
    if files is None:
        raise FileExistsError
    else:
        for file in files:
            image = face_recognition.load_image_file(join(_IMAGES_DIR, file))

            face_encoding = face_recognition.face_encodings(image)[0]

            file_json = {
                'name': splitext(file)[0],
                'encoding': face_encoding.tolist()
            }

            facial_encoding_data.append(file_json)

    return facial_encoding_data


def webcam_enable():
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():

    course_facial_data = _init_images()
    webcam_enable()

if __name__ == "__main__":
    main()
