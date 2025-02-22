import cv2
import dlib
import numpy as np
import os

# Load face detector dan face recognition model dari dlib
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Fungsi untuk mendapatkan encoding wajah
def get_face_encoding(image, face_rect):
    shape = sp(image, face_rect)  # Deteksi landmark wajah
    encoding = np.array(face_rec_model.compute_face_descriptor(image, shape))  # Ekstrak fitur wajah
    return encoding

# Fungsi untuk memuat wajah referensi dari folder dengan subfolder sebagai label
def load_reference_faces(folder="faceRecognition/faceRecognitionMarvel/wajah/"):
    reference_encodings = {}
    
    for person_name in os.listdir(folder):
        person_path = os.path.join(folder, person_name)

        if not os.path.isdir(person_path):  # Skip jika bukan folder
            continue

        encodings = []
        for filename in os.listdir(person_path):
            filepath = os.path.join(person_path, filename)
            image = cv2.imread(filepath)
            if image is None:
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                encoding = get_face_encoding(image, faces[0])
                encodings.append(encoding)

        if encodings:
            reference_encodings[person_name] = np.mean(encodings, axis=0)  # Gunakan rata-rata encoding

    return reference_encodings

# Load wajah referensi dari folder
reference_faces = load_reference_faces()

# Buka video sebagai input
cap = cv2.VideoCapture("video.mp4")
threshold = 0.75  # Ambang batas pengenalan wajah

# Simpan video hasil deteksi
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop jika video selesai

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_image = frame[y1:y2, x1:x2]

        if face_image.shape[0] == 0 or face_image.shape[1] == 0:
            continue

        encoding = get_face_encoding(frame, face)

        label = "Unknown"
        min_distance = float("inf")

        # Bandingkan dengan wajah referensi
        for name, ref_encoding in reference_faces.items():
            distance = np.linalg.norm(ref_encoding - encoding)
            if distance < threshold and distance < min_distance:
                min_distance = distance
                label = name  # Gunakan nama folder sebagai label

        # Gambar kotak dan teks
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Face Recognition Marvel", frame)
    out.write(frame)  # Simpan frame ke video

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Tekan 'q' untuk keluar
        break

cap.release()
out.release()
cv2.destroyAllWindows()
