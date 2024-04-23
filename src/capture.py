import cv2
import os

def create_directory(directory):
    """
    Create a directory if it doesn't exist.

    Parameters:
        directory (str): The path of the directory to be created.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == "__main__":
    video = cv2.VideoCapture(0)

    facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    count = 0

    nameID = str(input("Enter Your Name: ")).lower()

    path = '../Dataset/FaceData/raw/' + nameID  # Đường dẫn đúng đến thư mục raw

    create_directory(path)

    while True:
        ret, frame = video.read()
        faces = facedetect.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            count = count + 1
            image_path = f"{path}/User-{nameID}-{count}.jpg"
            print("Creating Images........." + image_path)
            cv2.imwrite(image_path, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imshow("WindowFrame", frame)
        cv2.waitKey(1)
        if count > 100:
            break

    video.release()
    cv2.destroyAllWindows()
