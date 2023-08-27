import csv
from datetime import datetime

import cv2
import face_recognition
import numpy as np

video_capture = cv2.VideoCapture(0)

# load known faces
anish_image = face_recognition.load_image_file("faces/anish.jpg")
anish_encoding = face_recognition.face_encodings(anish_image)[0]

lovish_image = face_recognition.load_image_file("faces/lovish.png")
lovish_encoding = face_recognition.face_encodings(lovish_image)[0]

known_face_encodings = [anish_encoding, lovish_encoding]
known_face_names = ["Anish Singh Butola", "Lovish Don"]

# list of expected students
students = known_face_names.copy()
face_locations = []
face_encoding = []

# get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# csv writter
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if (matches[best_match_index]):
            name = known_face_names[best_match_index]

            # add text if person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (100, 100)
                fontScale = 0.75
                fontColor = (100, 150, 36)
                thickness = 2
                lineType = 4
                cv2.putText(
                    frame, name + " - Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow(([name, current_time]))

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
