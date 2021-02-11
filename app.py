from flask import Flask, render_template
from flask import jsonify
from flask import session
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, emit
import os
import cv2
from scipy.spatial import distance 
from imutils import face_utils
import base64
import numpy as np
import dlib
from PIL import Image
import imageio
import pandas as pd
import math
import sklearn

# from mlxtend.image import extract_face_landmarks
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from io import BytesIO
import joblib
from engineio.payload import Payload

p = "./shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[15], mouth[21])
    C = distance.euclidean(mouth[12], mouth[18])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = distance.euclidean(eye[1], eye[4])
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance.euclidean(eye[0], eye[1])
    p += distance.euclidean(eye[1], eye[2])
    p += distance.euclidean(eye[2], eye[3])
    p += distance.euclidean(eye[3], eye[4])
    p += distance.euclidean(eye[4], eye[5])
    p += distance.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)

def mouth_over_eye(eye):
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    mouth_eye = mar/ear
    return mouth_eye

def average(y_pred):
    for i in range(len(y_pred)):
        if i % 240 == 0 or (i+1) % 240 == 0:
            pass
        else: 
            average = float(y_pred[i-1] +  y_pred[i] + y_pred[i+1])/3
            if average >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
    return y_pred

def model(landmarks,mean,std):
    features = pd.DataFrame(columns=["EAR","MAR","Circularity","MOE"])
    eye = landmarks[36:68]
    ear = eye_aspect_ratio(eye)
    mar = mouth_aspect_ratio(eye)
    cir = circularity(eye)
    mouth_eye = mouth_over_eye(eye)

    df = features.append({"EAR":ear,"MAR": mar,"Circularity": cir,"MOE": mouth_eye},ignore_index=True)

    if math.isnan(std["EAR"]) == False:
        df["EAR_N"] = (df["EAR"]-mean["EAR"])/ std["EAR"]
        df["MAR_N"] = (df["MAR"]-mean["MAR"])/ std["MAR"]
        df["Circularity_N"] = (df["Circularity"]-mean["Circularity"])/ std["Circularity"]
        df["MOE_N"] = (df["MOE"]-mean["MOE"])/ std["MOE"]
    Result = clf.predict(df)
    if Result == 1:
        Result_String = "Drowsy"
    else:
        Result_String = "Normal"

    return Result_String, df.values

def calibration(imgArr):
    if imgArr is None:
        return 1
    data = []
    index = 0
    for img in imgArr:
        # Converting the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(img, 1)
        if len(rects) > 0:
        # For each detected face, find the landmark.
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                data.append(shape)

    if len(data) > 0:
        features_test = []
        for d in data:
            eye = d[36:68]
            ear = eye_aspect_ratio(eye)
            mar = mouth_aspect_ratio(eye)
            cir = circularity(eye)
            mouth_eye = mouth_over_eye(eye)
            features_test.append([ear, mar, cir, mouth_eye])

        features_test = np.array(features_test)
        x = features_test
        y = pd.DataFrame(x,columns=["EAR","MAR","Circularity","MOE"])
        df_means = y.mean(axis=0)
        df_std = y.std(axis=0)
        session['df_means'] = df_means
        session['df_std'] = df_std

        return 0
    return -1

def live(img,mean,std):
    if img is not None:
        data = []
        result = []
        #image font style numbers:
        font  = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10,400)
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        # Converting the image to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get faces into webcam's image
        rects = detector(img, 1)

        # For each detected face, find the landmark.
        for (i, rect) in enumerate(rects):
            # Make the prediction and transfom it to numpy array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            Result_String, features = model(shape,mean,std)
            data.append (features)
            result.append(Result_String)
            cv2.putText(img,Result_String, bottomLeftCornerOfText, font, fontScale, fontColor,lineType)

            # Draw on our image, all the finded cordinate points (x,y) 
            for (x, y) in shape:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        return img,result
    return

app = Flask(__name__)
app.secret_key = "root"
Payload.max_decode_packets = 500
socketio = SocketIO(app, cors_allowed_origins="*", async_handlers=True)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/")
@cross_origin()
def hello():
    calibrated = False
    session['inputImages'] = []
    session['calibrated'] = False
    session['sec'] = 0
    session['result'] = []

    return render_template('index.html', title='Face recognition')

@socketio.on('connect')
def test_connect():
    emit('logs', {'data': 'Connection established'})

@socketio.on('image')
def image(data_image):
    sbuf = StringIO()
    sbuf.write(data_image)

    debug = -1
    # decode and convert into image
    b = BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)

    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    # Process the image frame
    #put first 5 images to array for calibration
    if len(session['inputImages']) < 3 and session['sec'] > 3:
        session['inputImages'].append(frame)

    session['sec'] += 1
    #calculate ear,mar,puc,moe
    if len(session['inputImages']) == 3 and debug != 0:
        inputImages = session['inputImages']
        debug = calibration(inputImages)
        if debug == 0:
            emit('response_plot',session['df_std'].to_dict())
        session['calibrated'] = True
    #clear data if input fail
    if debug == -1 and session['calibrated'] == True:
        session['calibrated'] = False
        emit('response_image','pop success')
        if session['inputImages'] is not None:
            session['inputImages'].pop()
        session['inputImages'] = []

    #view prediction
    resultFrame = frame
    if debug == 0:
        resultFrame,session['result'] = live(frame,session['df_means'],session['df_std'])
        emit('response_status',session['result'])

    scale_percent = 100 # percent of original size
    width = int(resultFrame.shape[1] * scale_percent / 100)
    height = int(resultFrame.shape[0] * scale_percent / 100)
    dim = (width, height)
# resize image
    resized = cv2.resize(resultFrame, dim, interpolation = cv2.INTER_AREA)

    imgencode = cv2.imencode('.jpg', resized)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

if __name__ == "__main__":
    try:
        clf = joblib.load("logistic_model.pkl")
        print ('Model loaded')
        print('new code 3')
    except:
        print("error loading model")
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    socketio.run(app,host="0.0.0.0", port=os.environ['PORT'], ssl_context=('cert.pem', 'key.pem'))