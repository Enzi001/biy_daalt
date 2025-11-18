import opencv_test
from pathlib import Path

# Locate the Haar Cascade file inside OpenCV installation
cascade_path = Path(opencv_test.data.haarcascades) / "haarcascade_frontalface_default.xml"

# Load the classifier
clf = opencv_test.CascadeClassifier(str(cascade_path))

# Open webcam
camera = opencv_test.VideoCapture(0, opencv_test.CAP_DSHOW)  # CAP_DSHOW avoids some Windows errors

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to access camera!")
        break
    
    gray = opencv_test.cvtColor(frame, opencv_test.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        opencv_test.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

    opencv_test.imshow("Live Face Detection - Press Q to Quit", frame)

    if opencv_test.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
opencv_test.destroyAllWindows()