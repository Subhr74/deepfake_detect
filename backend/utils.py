import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_list = []
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face_list.append((face, (x, y, w, h)))

    return face_list