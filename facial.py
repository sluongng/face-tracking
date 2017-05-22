import face_recognition
import cv2
import json

from os import listdir, getcwd
from os.path import isfile, join, dirname, abspath

_IS_VERBOSE = 0

if _IS_VERBOSE:
    print("Initiating facial.py !!!")
    print("Face_Recognition version is:", face_recognition.__version__)
    print("OpenCV version is:", cv2.__version__)

_WORKING_DIR = getcwd()
_IMAGES_DIR = join(_WORKING_DIR, "images")

out_json = []

if _IS_VERBOSE:
    print("Working directory is:", _WORKING_DIR)
    print("Images directory is:", _IMAGES_DIR)

files = [f for f in listdir(_IMAGES_DIR) if isfile(join(_IMAGES_DIR, f))]
if files is None:
    raise FileExistsError
else:
    for file in files:
        image = face_recognition.load_image_file(join(_IMAGES_DIR, file))

        face_encoding = face_recognition.face_encodings(image)[0]

        file_json = {
            'name': file,
            'encoding': face_encoding.tolist()
        }

        out_json.append(file_json)

result = json.dumps(out_json, sort_keys=True, separators=(',', ': '), indent=4)
print(result)
