import face_recognition
import cv2
# import json
# import numpy as np

from os import listdir, getcwd
from os.path import isfile, join, splitext

_IS_VERBOSE = 0

if _IS_VERBOSE:
    print("Initiating facial.py !!!")
    print("Face_Recognition version is:", face_recognition.__version__)
    print("OpenCV version is:", cv2.__version__)

_WORKING_DIR = getcwd()
_IMAGES_DIR = join(_WORKING_DIR, "images")

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


def webcam_enable(course_facial_data):
    cap = cv2.VideoCapture(0)

    process_this_frame = True

    while True:
        ret, frame = cap.read()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            face_names = []
            test_encoding = course_facial_data[0]['encoding']

            for face_encoding in face_encodings:
                match = face_recognition.compare_faces([test_encoding], face_encoding)
                name = 'Unknown'

                if match[0]:
                    name = course_facial_data[0]['name']

                face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    course_facial_data = _init_images()

    webcam_enable(course_facial_data)

if __name__ == "__main__":
    main()
