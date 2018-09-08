import sys
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi
import pandas as pd
import matplotlib.pyplot as plt
import csv
from keras.models import load_model
from keras.preprocessing import image
import os, time, datetime
import numpy as np
import threading
import tensorflow as tf
import matplotlib.image as mpimg

class caption(QDialog):
    def __init__(self):
        super(caption, self).__init__()
        loadUi('main.ui',self)
        def model_load():
            global graph
            self.output.setText("Loading Model...")
            self.classifier = load_model("model_86.h5")
            graph = tf.get_default_graph()
            self.output.setText("Model Successfully loaded.")
            self.modelLoad.setFlat(True)
            self.modelLoad.setText("")

        self.Image = None
        self.classifier = None
        self.t = threading.Thread(target=model_load)
        self.startButton.clicked.connect(self.start_webcam)
        self.modelLoad.clicked.connect(self.t.start)
        self.analyse.clicked.connect(self.guess)
        self.today.clicked.connect(self.plot)
        self.stress.clicked.connect(self.stress_plot)


    def plot(self):
        self.output.setText("Getting the Statistics...")
        file = "./log/"+str(datetime.datetime.now())[:10]+".csv"
        try:
            dataset = pd.read_csv(file)
            fear = dataset.iloc[:,2].values
            ang = dataset.iloc[:,3].values
            dis = dataset.iloc[:,4].values
            hap = dataset.iloc[:,5].values
            neu = dataset.iloc[:,6].values
            sad = dataset.iloc[:,7].values
            sur = dataset.iloc[:,8].values

            heights = [sum(fear)*100/(max(len(fear),1)),sum(ang)*100/(max(len(ang),1)),sum(dis)*100/(max(len(dis),1)),sum(hap)*100/(max(len(hap),1)),sum(neu)*100/(max(len(neu),1)),sum(sad)*100/(max(len(sad),1)),sum(sur)*100/(max(len(sur),1))]
            labels = ['Fear','Angry','Disgust','Happy','Neutral','Sad','Surprised']
            plt.bar(list(range(7)), heights, align='center')
            plt.xticks(list(range(7)),labels)
            plt.xlabel('Expressions')
            plt.ylabel('Percentage')
            plt.title("About today")
            plt.show()
            self.output.clear()
            
        except FileNotFoundError:
            self.output.setText("Sorry, no Statistics available.")

    def stress_plot(self):
        self.output.setText("Getting Stress levels...")
        file = "./log/"+str(datetime.datetime.now())[:10] +".csv"
        try:
            dataset = pd.read_csv(file)
            fear = dataset.iloc[:,2].values
            ang = dataset.iloc[:,3].values
            dis = dataset.iloc[:,4].values
            hap = dataset.iloc[:,5].values
            neu = dataset.iloc[:,6].values
            sad = dataset.iloc[:,7].values
            sur = dataset.iloc[:,8].values

            heights = [(sum(fear)+sum(ang)+sum(sad))*100/max(len(fear),len(sad),len(ang),1),(sum(dis)+sum(hap)+sum(sur)+sum(neu))*100/max(len(dis),len(hap),len(sur),len(neu),1)]
            labels = ['Stressed Emotions','Non-Stressed Emotions']
            plt.bar(list(range(2)), heights, align='center')
            plt.xticks(list(range(2)),labels)
            plt.ylabel('Stress Percentage')
            plt.title("About today")
            plt.show()
            self.output.clear()
            
        except FileNotFoundError:
            self.output.setText("Sorry, no Statistics available.")


    def guess(self):
        global graph
        self.timer.stop()
        cascPath = "haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        date = str(datetime.datetime.now())[:10]

        camera = cv2.VideoCapture(1)
        retval, im = camera.read()
        
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                                 minNeighbors=5,
                                                 minSize=(30, 30),
                                                 flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite("./cam/test.png",im)
            sub_face = im[y:y + h, x:x + w]
            face_file_name = "detected_faces/face.jpg"
            cv2.imwrite(face_file_name, sub_face)
            break
        emo = ""
        test_img = image.load_img("detected_faces/face.jpg", target_size=(128, 128))
        test_img = image.img_to_array(test_img)
        test_img = np.expand_dims(test_img, axis=0)
        with graph.as_default():
            result = self.classifier.predict(test_img)
        print(result)

        if (result[0][0] == 1):
            emo = "Fear"
            plt.title(emo)
            print(emo)
        if (result[0][1] == 1):
            emo = "Angry"
            plt.title(emo)
            print(emo)
        if (result[0][2] == 1):
            emo = "Disgust"
            plt.title(emo)
            print(emo)
        if (result[0][3] == 1):
            emo = "Happy"
            plt.title(emo)
            print(emo)
        if (result[0][4] == 1):
            emo = "Neutral"
            plt.title(emo)
            print(emo)
        if (result[0][5] == 1):
            emo = "Sad"
            plt.title(emo)
            print(emo)
        if (result[0][6] == 1):
            emo = "Surprised"
            plt.title(emo)
            print(emo)
        else:
            emo = "Please try again."

        with open(".//log//" + date + ".csv", "a", newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([str(datetime.datetime.now()), emo, result[0][0], result[0][1],
                                 result[0][2], result[0][3], result[0][4], result[0][5], result[0][6]])
        
        img = mpimg.imread('./cam/test.png')
        imgplot = plt.imshow(img)
        plt.tick_params(axis='both', which='both',bottom='off',top='off',labelbottom='off',right='off',left='off',labelleft='off')
        plt.show()
       


    def start_webcam(self):
        self.capture = cv2.VideoCapture(1)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(5)


    def update_frame(self):
        ret,self.Image = self.capture.read()
        #ret,self.Image = cv2.VideoCapture(1).read()
        self.Image  = cv2.flip(self.Image,1)
        self.displayImage(self.Image,1)
       

    def displayImage(self,img,window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3 :
            if img.shape[2] == 4 :
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img,img.shape[1],img.shape[0],img.strides[0],qformat)
        #BGR>RGB
        outImage = outImage.rgbSwapped()

        if window == 1:
           self.imglabel.setPixmap(QPixmap.fromImage(outImage))
           self.imglabel.setScaledContents(True)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = caption()
    window.setWindowTitle("Facial")
    window.show()
    sys.exit(app.exec_())