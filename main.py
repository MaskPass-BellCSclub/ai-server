from flask import Flask, Response, request
from flask_cors import CORS
import threading
import time
from tensorflow.python.keras.models import load_model
from tensorflow.compat.v1 import InteractiveSession  # type: ignore
from tensorflow.compat.v1 import ConfigProto  # type: ignore
import numpy as np
import cv2
import urllib

# Global Flags
inFrame = None
finishedFrame = None
should_open_door = False
stop_ai_server = False
stop_camera_server = False

# Get camera from mjpg stream
def openIpCam(ip):
    global inFrame
    global stop_camera_server
    while True:
        try:
            camera = cv2.VideoCapture(ip)
            while True:
                if stop_camera_server:
                    print("STOPPING CAMERA SERVER")
                    stop_camera_server = False
                    camera.release()
                    inFrame = None
                    return
                # Auto-reconnects if camera connection dies
                if camera.isOpened():
                    ret, frame = camera.read()
                    if frame is None:
                        camera.release()
                    else:
                        inFrame = frame
                else:
                    print("Failed To Open Ip Camera " + str(ip) + "!")
                    camera.release()
                    camera = cv2.VideoCapture(ip)
                    time.sleep(0.5)
        except Exception as e:
            print(f"Open Camera Error: {e}")
        time.sleep(0.5)

# Runs AI thread
def categorize():
    global should_open_door
    global finishedFrame
    global inFrame
    global stop_ai_server

    config = ConfigProto()
    # allow growth
    config.gpu_options.allow_growth = True

    keras_model = load_model("./mask_model.h5")
    classifier = cv2.CascadeClassifier(f"{cv2.haarcascades}/haarcascade_frontalface_alt2.xml")

    label = {
        0: {"name": "Mask on chin", "color": (51, 153, 255), "id": 0},
        1: {"name": "Mask below nose", "color": (255, 255, 0), "id": 1},
        2: {"name": "No mask", "color": (0, 0, 255), "id": 2},
        3: {"name": "Mask on", "color": (0, 102, 51), "id": 3},
    }

    maskOnBuffer = 0

    while True:
        if stop_ai_server:
            print("STOPPING AI SERVER")
            stop_ai_server = False
            return 

        if inFrame is None:
            continue

        frame = inFrame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = classifier.detectMultiScale(gray)

        for x, y, w, h in detections:
            color = (0, 0, 0)
            gray_face = gray[y:y+h+50,x:x+w+50]

            # Check location
            if gray_face.shape[0] >= 200 and gray_face.shape[1] >= 200:

                gray_face = cv2.resize(gray_face, (300, 300))
                gray_face = gray_face / 255
                gray_face = np.expand_dims(gray_face, axis=0)
                gray_face = gray_face.reshape((1, 300, 300, 1))
                pred = np.argmax(keras_model.predict(gray_face))
                classification = label[pred]["name"]
                color = label[pred]["color"]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, label[pred]["id"])

                if classification == "Mask on":
                    # print("Mask On", maskOnBuffer)
                    maskOnBuffer += 1
                    if maskOnBuffer > 20:
                        # print("Allow In")
                        should_open_door = True
                else:
                    maskOnBuffer = 0
                    should_open_door = False

                cv2.putText(
                    frame,
                    classification,
                    (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                if classification == "No mask":
                    cv2.putText(
                        frame,
                        "Please Use a Mask!",
                        (20, 20 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                elif classification == "Mask on":
                    cv2.putText(
                        frame,
                        "Thank You!",
                        (20, 20 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 102, 51),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Improper Mask!",
                        (20, 20 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (51, 153, 255),
                        2,
                        cv2.LINE_AA,
                    )
                break
        else:
            cv2.putText(
                frame,
                "Looking For Face...",
                (20, 20 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        finishedFrame = frame


# Camera API service
cameraAPI = Flask(__name__)
CORS(cameraAPI)

cameraAPI.route("/", methods=["GET"], strict_slashes=False)

@cameraAPI.route("/", methods=["GET"])
def index():
    return Response(status=200)

@cameraAPI.route("/video_feed", methods=["GET"])
def video_feed():
    return Response(generate_next_camera_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@cameraAPI.route("/start_ai", methods=["POST"])
def start_ai():
    content = request.get_json(silent=True)

    # Create and start ip camera thread
    IPCamThread = threading.Thread(target=openIpCam, args=(content["camera"], ))
    IPCamThread.setDaemon(True)
    IPCamThread.start()

    # Create and start ai engine thread
    aiThread = threading.Thread(target=categorize)
    aiThread.setDaemon(True)
    aiThread.start()

    return Response(status=200)

@cameraAPI.route("/open_door", methods=["GET"])
def open_doors():
    global should_open_door
    return str(should_open_door)

@cameraAPI.route("/stop", methods=["GET"])
def stop():
    global stop_ai_server
    global stop_camera_server
    stop_ai_server = True
    stop_camera_server = True
    return Response(status=200)

@cameraAPI.route("/debug", methods=["GET"])
def debug():
    return str(finishedFrame)

def generate_next_camera_frame():
    while True:
        time.sleep(0.05)
        _, jpg = cv2.imencode('.jpg', finishedFrame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        # _, jpg = cv2.imencode('.jpg', finishedFrame)
        frame = jpg.tobytes()
        yield (b'--frame\r\n'
           b'Content-Type:image/jpeg\r\n'
           b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
           b'\r\n' + frame + b'\r\n')

# start the server lol
cameraAPI.run(host="0.0.0.0", port=5000)
