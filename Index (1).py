from flask import Flask, render_template, request
from flask_mysqldb import MySQL

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'project'
mysql = MySQL(app)

@app.route('/')
def index():
  return render_template('Index.html')

@app.route('/MusicPlayer')
def message():
    import cv2
    import numpy as np
    from keras.models import model_from_json

    # Emotion Categories
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Load json File and Create Model
    json_file = open("Models\\emotion_model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # Load Weights into New Model
    emotion_model.load_weights("Models\\emotion_model.h5")

    # Capturing Image
    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # while True:
    #     result, img = cap.read()
    #     cv2.imshow("Camera",img)
    #     flag = cv2.waitKey(1)
    #     if flag == ord("\r"):
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         break
    #     elif flag == ord("\b"):
    #         cap.release()
    #         cv2.destroyAllWindows()
    #         exit()

    img = cv2.imread("Test\\Surprised.jpeg")

    # Face Detection using Haar Cascade
    frame = img
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Emotion Detection
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)

    # Displaying Emotions
    # cv2.imshow("Emotion State", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Saving Image for Training
    # destiny = "Train\\"+str(emotion_dict[maxindex])+"\\"+str(x+y+w+h)+".jpg"
    # cv2.imwrite(destiny,img)

    cursor = mysql.connection.cursor()

    # Executing SQL Statements
    cursor.execute("Select Name from playlist where Emotion=\'"+emotion_dict[maxindex]+"\';")
    n = cursor.fetchall()
    cursor.execute("Select Description from playlist where Emotion=\'"+emotion_dict[maxindex]+"\';")
    d = cursor.fetchall()
    cursor.execute("Select Icon from playlist where Emotion=\'"+emotion_dict[maxindex]+"\';")
    i = cursor.fetchall()
    cursor.execute("Select Link from playlist where Emotion=\'"+emotion_dict[maxindex]+"\';")
    l = cursor.fetchall()

    # Closing the cursor
    cursor.close()

    return render_template('Player.html',emo=emotion_dict[maxindex], n=n, d=d, i=i, l=l, m=maxindex)


if __name__ == '__main__':
    app.run(debug=True)