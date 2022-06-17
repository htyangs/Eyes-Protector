
from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import QTimer 
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import (QApplication, QMessageBox, )
import sqlite3
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates 
import math
import eye_ui as ui
import requests

class Window(QDialog, ui.Ui_Dialog):
    def __init__(self):
        super().__init__()

        self.token = ''
        self.camera = cv.VideoCapture(0)
        self.setupUi(self)
        self.user_list.currentTextChanged.connect(self.user_list_onchange)
        self.user_list_2.currentTextChanged.connect(lambda: self.user_list_onchange(2))
        self.confirm_push.clicked.connect(self.confirm_push_onchange)
        self.add_push.clicked.connect(self.add_push_onchange)
        #self.user_list.addItem('None')
        self.blink_threshold.valueChanged.connect(self.blink_threshold_onchange)
        self.bright_threshold.valueChanged.connect(self.bright_threshold_onchange)
        self.distance_threshold.valueChanged.connect(self.distance_threshold_onchange)

        self.blink_bar.valueChanged.connect(self.blink_bar_onchange)
        self.bright_bar.valueChanged.connect(self.bright_bar_onchange)
        self.distance_bar.valueChanged.connect(self.distance_bar_onchange)

        self.working_time.valueChanged.connect(self.working_time_onchange)
        self.resting_time.valueChanged.connect(self.resting_time_onchange)
        self.start_push.clicked.connect(self.start_push_onchange)
        self.initialize_push.clicked.connect(self.initialize_push_onchange)
        self.want_line.clicked.connect(self.want_line_onchange)
        
        #timer
        self.timer_camera = QTimer() #初始化定时器
        self.timer_warm = QTimer() #初始化定时器
        self.timer_camera.timeout.connect(self.update_progress_value)
        self.timer_warm.timeout.connect(self.check_status)
        self.tabWidget.currentChanged.connect(self.change_index)
        #self.main.tabBarClicked.connect(self.pushButton_func,0)
        #self.analyze.tabBarClicked.connect(self.pushButton_func,1)
        self.work_time = self.working_time.value()
        self.rest_time = self.resting_time.value()
        self.blink_thres = self.blink_threshold.value()
        self.bright_thres = self.bright_threshold.value()
        self.distance_thres = self.distance_threshold.value()
        self.exercise.addItem('None')
        self.exercise.addItem('close eye')
        self.exercise.addItem('jumping jack')
        # variables 
        self.FONT_SIZE = 1
        # calendar
        self.select_range.addItem('Every Minute')
        self.calendarWidget.selectionChanged.connect(self.calendar)

        self.frame_counter =0
        self.CEF_COUNTER =0
        self.total_blink =0
        self.eye_area= 800
        self.ratio = 0
        self.count = 0
        self.brightness_value = 0
        # constants
        self.eye_close_frame =1
        self.previous_time = 200
        self.area_record = np.ones(self.previous_time)
        self.FONTS =cv.FONT_HERSHEY_COMPLEX
        self.EYE_STATE = 0
        self.ratio_thres = 4.5
        self.eye_area_thres_high = 1500
        self.eye_area_thres_low = 200
        self.eye_area_record = 800
        self.eye_area_ratio = 0.7
        # face bounder indices 
        self.FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]
        self.FACE_OVAL_SIM = [156,383,397]
        # lips indices for Landmarks
        self.LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
        self.LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
        # Left eyes indices 
        self.LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
        self.LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
        # right eyes indices
        self.RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
        self.RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]
        # Center
        self.CENTER_POINT = [9,8,168]
        self.BODY = [22,20,18,16,14,12,24,23,11,13,15,17,19,21]
        self.HEAD = [8,6,5,4,0,1,2,3,7]
        self.map_face_mesh = mp.solutions.face_mesh
        self.status = 'run' # start # end
        self.blink_counter = 0
        self.area_counter = 0
        self.bright_counter = 0
        self.frame_counter = 0
        self.passing_time = 0

        #store minute information
        self.count_minute = 0 
        self.previous_minute = 0
        self.count_bright = 0
        self.count_blink = 0
        self.count_distance = 0

        #record time
        self.previous_time_step = 0
        self.now_time_step = 0
        self.pass_time = 0 
        self.time_status = 'start'

        #jump
        self.previous_state = -1
        self.count_hand = 0
        self.count_jump = 0
        self.shoulder_pos = []
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.current_user  = str(self.user_list.currentText())
        self.con = sqlite3.connect('database.db')
        self.cursorObj = self.con.cursor()
        self.cursorObj.execute('create table if not exists None(year, month, day, hour, minute, distance, brightness, blink, state)')
        self.cursorObj.execute('create table if not exists threshold(user, line_token,distance_area,distance_ratio, brightness, blink,UNIQUE(user)) ')        
        self.cursorObj.execute("insert or ignore into threshold(user,line_token, distance_area,distance_ratio, brightness, blink) VALUES (?,?,?,?,?,?)" ,('None','',self.eye_area_record,self.eye_area_ratio,60,4))
        self.con.commit() 
        cursor = self.cursorObj.execute("SELECT * from threshold").fetchall()
        for row in cursor:
            self.user_list.addItem(row[0])
            self.user_list_2.addItem(row[0])
            print(row)
        self.con.commit() 
        
    def __del__(self):
        self.update_database()
        self.summary_report()
        self.connection.close()
    def closeEvent(self, event):
        self.summary_report()

    def lineNotifyMessage(self,msg):
        try:
            headers = {
                "Authorization": "Bearer " + self.token, 
                "Content-Type" : "application/x-www-form-urlencoded"
            }
            
            payload = {'message': msg}
            r = requests.post("https://notify-api.line.me/api/notify", headers = headers, params = payload)
        except:
            pass
    def summary_report(self):
        year = datetime.today().strftime("%Y")
        month =  datetime.today().strftime("%m")
        day =  datetime.today().strftime("%d")    
        today_date =  datetime.today().strftime("%Y-%m-%d")    
        print(year,month,day)   
        self.cursorObj = self.con.cursor()
        #self.cursorObj.execute("SELECT EXISTS(SELECT 1 FROM threshold WHERE user=? LIMIT 1)", ('default',))
        cursor = self.cursorObj.execute("SELECT year, month, day, hour, minute, distance, brightness, blink, state  from %s WHERE year= %s AND month=%s AND day=%s " %(self.current_user,year,month,day,))
        self.con.commit() 
        #print(cursor.fetchall())
        date = []
        dis = []
        bri = []
        blink = []
        use = [] 
        for i in cursor:
            use.append(i[8])
            dis.append(float(i[5]))
            bri.append(int(i[6]))
            blink.append(int(i[7]))
        if (len(use)!=0):
            use_time = use.count(2)
            not_time = use.count(1)
            rest_time = use.count(0)
            avg_dis = sum(dis) / len(dis)
            avg_bri = sum(bri) / len(bri)
            avg_blink = sum(blink) / len(blink)
            if (self.want_line.isChecked()):
                self.lineNotifyMessage(today_date+'\n use time : '+ str(use_time) +' minutes \n absent time : '+str(not_time)+ ' minutes \n rest time : '+str(rest_time)+' minutes' \
                +'\n average distance : '+ str(avg_dis) +' \n average brigntness : '+str(avg_bri)+ ' \n average blink : '+str(avg_blink))
            print(' use time : ', str(use_time) ,'\n absent time',str(not_time), '\n rest time',str(rest_time))
            print(' average distance: ', avg_dis ,'\n average brigntness',avg_bri, '\n average blink',avg_blink)
    def want_line_onchange(self):
        if (self.want_line.isChecked()):
            self.line_token.setEnabled(True)
        else:
            self.line_token.setEnabled(False)
                

    def change_index(self,value):
        self.stackedWidget.setCurrentIndex(value)
    
    def user_list_onchange(self,user = 1):
        self.update_database()
        self.current_user  = str(self.user_list.currentText())
        if user ==2 :
            self.current_user  = str(self.user_list_2.currentText())
            self.calendar()
        self.con = sqlite3.connect('database.db')
        self.cursorObj = self.con.cursor()
        cursor = self.cursorObj.execute("SELECT user,line_token, distance_area,distance_ratio, brightness, blink  from threshold  WHERE user = '%s'" %(self.current_user,))
        self.con.commit() 
        for row in cursor:
            self.blink_threshold.setValue(float(row[5]))
            self.bright_threshold.setValue(float(row[4]))
            self.distance_threshold.setValue(float(row[3]))
            self.eye_area_record = (float(row[2]))
            self.token = row[1]
        self.con.commit() 

    def add_user_onchange(self):
        pass

    def confirm_push_onchange(self):
        self.initialize_push.setEnabled(True)
        self.start_push.setEnabled(True)
        self.line_token.setText(self.token)
        self.start_time = time.time()
        self.status = 'run'
        self.timer_camera.start(5)
        self.timer_warm.start(30)

    def calendar(self):
        selectDay = self.calendarWidget.selectedDate()
        year = selectDay.toString("yyyy")
        month =  selectDay.toString("M")
        day =  selectDay.toString("d")
        print(year,month,day)

        self.cursorObj = self.con.cursor()
        #self.cursorObj.execute("SELECT EXISTS(SELECT 1 FROM threshold WHERE user=? LIMIT 1)", ('default',))
        cursor = self.cursorObj.execute("SELECT year, month, day, hour, minute, distance, brightness, blink, state  from %s WHERE year= %s AND month=%s AND day=%s " %(self.current_user,year,month,day,))
        self.con.commit() 
        #print(cursor.fetchall())
        date = []
        dis = []
        bri = []
        blink = []
        use = []
        for i in cursor:
            date.append(datetime(i[0], i[1], i[2], i[3],i[4]))
            use.append(i[8])
            dis.append(float(i[5]))
            bri.append(int(i[6]))
            blink.append(int(i[7]))
        print(date)
        xfmt = matplotlib.dates.DateFormatter('%H:%M')
        datestime = matplotlib.dates.date2num(date)
        print((datestime, dis))
        plt.gca().xaxis.set_major_formatter(xfmt)
        plt.plot_date(datestime, use,linestyle='solid')
        plt.ylim(-0.1,2.1)
        plt.title('Using Time')
        plt.savefig('use.png')
        plt.close()

        plt.gca().xaxis.set_major_formatter(xfmt)
        self.display_image(cv.imread('use.png'),(400,270),self.use_time_graph )
        plt.plot_date(datestime, dis,linestyle='solid')
        plt.ylim(0,2)
        plt.title('Distance')
        plt.savefig('dis.png')
        plt.close()

        plt.gca().xaxis.set_major_formatter(xfmt)
        self.display_image(cv.imread('dis.png'),(400,270),self.distance_graph )
        plt.plot_date(datestime, bri,linestyle='solid')
        plt.ylim(0,255)
        plt.title('Brightness')
        plt.savefig('bri.png')
        plt.close()

        plt.gca().xaxis.set_major_formatter(xfmt)
        self.display_image(cv.imread('bri.png'),(400,270),self.brightness_graph )
        plt.plot_date(datestime, blink,linestyle='solid')
        plt.ylim(0,60)
        plt.title('Blinking')
        plt.savefig('blink.png')
        plt.close()
        self.display_image(cv.imread('blink.png'),(400,270),self.blink_graph )

    def display_image(self,img,size,target):
        show = cv.resize(img,size)
        #show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
        target.setPixmap(QPixmap.fromImage(showImage))    

    def add_push_onchange(self):
        text = str(self.add_user.text())
        self.user_list.addItem(text)
        self.user_list_2.addItem(text)
        self.add_user.clear()
        if (text != ''):
            self.con = sqlite3.connect('database.db')
            self.cursorObj = self.con.cursor()
            try:
                self.cursorObj.execute('create table if not exists %s (year, month, day, hour, minute, distance, brightness, blink, state)' %(text))
                self.cursorObj.execute("insert or ignore into threshold(user,line_token,  distance_area, distance_ratio, brightness, blink) VALUES (?,?,?,?,?,?)" ,(text,self.line_token.text(),self.eye_area_record,self.eye_area_ratio,60,4))
                self.con.commit()
            except:
                self.showDialog('Not valid name!')
        else:
            print('empty')


    def working_time_onchange(self):
        self.work_time = self.working_time.value()

    def resting_time_onchange(self):
        self.rest_time = self.resting_time.value()

    def initialize_push_onchange(self):

        br = self.ratio*1.18
        bv = self.brightness_value*0.65
        dis = 0.7
        self.blink_threshold.setValue(br)
        self.bright_threshold.setValue(bv)
        self.distance_threshold.setValue(dis)
        self.eye_area_record = self.eye_area
        self.update_database()
    def update_database(self):
        self.cursorObj.execute("UPDATE threshold SET distance_area = %s, distance_ratio = %s ,  brightness= %s , blink=%s  WHERE user='%s'" %(self.eye_area, self.distance_threshold.value(),self.bright_threshold.value(),self.blink_threshold.value(),self.current_user))
        self.con.commit()   

    def start_push_onchange(self):
        self.counter = -1
        self.pass_time = 0.01
        if (self.want_line.isChecked()):
            self.lineNotifyMessage('start')
        self.status = 'start'
        self.time_status = 'work'
        self.previous_minute = 0
        self.init_time = time.time()
        self.previous_time_step = time.time()

    def blink_threshold_onchange(self):
        self.blink_thres = self.blink_threshold.value()
        self.blink_bar.setValue(int(self.blink_thres*10))
    def blink_bar_onchange(self):
        self.blink_thres = self.blink_bar.value()/10
        self.blink_threshold.setValue(self.blink_thres)
        #self.update_database()
    def bright_threshold_onchange(self):
        self.bright_thres = self.bright_threshold.value()
        self.bright_bar.setValue(int(self.bright_thres))
    def bright_bar_onchange(self):
        self.bright_thres = self.bright_bar.value()
        self.bright_threshold.setValue(self.bright_thres)

    def distance_threshold_onchange(self):
        self.distance_thres = self.distance_threshold.value()
        self.distance_bar.setValue(int(self.distance_thres*100))
    def distance_bar_onchange(self):
        self.distance_thres = self.distance_bar.value()/100
        self.distance_threshold.setValue(self.distance_thres)
        #self.update_database()
    def check_status(self):
        if (self.status == 'start'):
            if(self.area_counter>2):
                self.showDialog('Too close',line=True)
                self.area_counter = 0
            if(self.bright_counter >20):
                #self.showDialog('Too dim')
                self.bright_counter = 0
    ''' eye detection function '''

    def PolyArea(self,x,y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    # landmark detection function 
    def landmarksDetection(self,img, results, draw=False,body=False):
        img_height, img_width= img.shape[:2]
        # list[(x,y), (x,y)....]
        if(body==False):
            mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
        else:
            mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.pose_world_landmarks.landmark]
        if draw :
            [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]
        # returning the list of tuples for each landmarks 
        return mesh_coord

    # Euclaidean distance 
    def euclaideanDistance(self,point, point1):
        x, y = point
        x1, y1 = point1
        distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
        return distance

    def blinkRatio(self,img, landmarks, right_indices, left_indices):
        # Right eyes 
        # horizontal line 
        rh_right = landmarks[right_indices[0]]
        rh_left = landmarks[right_indices[8]]
        # vertical line 
        rv_top = landmarks[right_indices[12]]
        rv_bottom = landmarks[right_indices[4]]
        # draw lines on right eyes 
        # LEFT_EYE 
        # horizontal line 
        lh_right = landmarks[left_indices[0]]
        lh_left = landmarks[left_indices[8]]
        # vertical line 
        lv_top = landmarks[left_indices[12]]
        lv_bottom = landmarks[left_indices[4]]
        rhDistance = self.euclaideanDistance(rh_right, rh_left)
        rvDistance = self.euclaideanDistance(rv_top, rv_bottom)
        lvDistance = self.euclaideanDistance(lv_top, lv_bottom)
        lhDistance = self.euclaideanDistance(lh_right, lh_left)
        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance
        ratio = (reRatio+leRatio)/2
        return ratio 


    def get_average_brightness(self,image,mesh_coords,frame_height,frame_width):
        lum = image[:,:,0]*0.144+image[:,:,1]*0.587+image[:,:,2]*0.299
        vals = np.average(lum)
        if math.isnan (vals) :
            return 0
        else:
            return vals

    def colorBackgroundText(self,img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
        (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # getting the text size
        x, y = textPos
        cv.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
        cv.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

        return img


    def showDialog(self,text,line=True):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(text)
        msgBox.setWindowTitle("Warning")
        msgBox.exec()
        if (line and self.want_line.isChecked()):
            self.lineNotifyMessage(text)

    def get_state_body(self,results):
        up_state = 1
        down_state = -1
        left_wrist = results.pose_world_landmarks.landmark[15]
        left_pinky = results.pose_world_landmarks.landmark[17]
        right_wrist = results.pose_world_landmarks.landmark[16]
        right_pinky =  results.pose_world_landmarks.landmark[18]
        left_hip = results.pose_world_landmarks.landmark[23]
        right_hip = results.pose_world_landmarks.landmark[24]
        nose = results.pose_world_landmarks.landmark[0]
        if(left_wrist.y < nose.y and right_wrist.y < nose.y):
            return up_state
        elif(left_pinky.y > left_hip.y and right_pinky.y > right_hip.y):
            return down_state
        return 0

    def update_progress_value(self):
        try:
            if(self.status != 'rest'):
                with self.map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
                    self.frame_counter += 1
                    ret, frame = self.camera.read() 
                    frame_height, frame_width= frame.shape[:2]
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    results  = face_mesh.process(rgb_frame)
                    FONT = cv.FONT_HERSHEY_COMPLEX
                    
                    if results.multi_face_landmarks:
                        self.record_state = 2
                        if (self.time_status == 'work'):
                            #print(time.time() , self.previous_time_step)
                            self.pass_time += (time.time() - self.previous_time_step)
                            self.previous_time_step =  time.time()
                        else:
                            self.pass_time += 0
                        if(self.status == 'start'):
                            self.time_status = 'work'
                        mesh_coords = self.landmarksDetection(frame, results, False)
                        right_eye_area = self.PolyArea(np.array([mesh_coords[p] for p in self.RIGHT_EYE ])[:,0],np.array([mesh_coords[p] for p in self.RIGHT_EYE ])[:,1])
                        left_eye_area = self.PolyArea(np.array([mesh_coords[p] for p in self.LEFT_EYE ])[:,0],np.array([mesh_coords[p] for p in self.LEFT_EYE ])[:,1])
                        self.eye_area = (right_eye_area+left_eye_area)/2
                        self.ratio = self.blinkRatio(frame, mesh_coords, self.RIGHT_EYE, self.LEFT_EYE)
                        self.brightness_value = self.get_average_brightness(rgb_frame,mesh_coords,frame_height,frame_width) 
                        #area_record[counter] = eye_area 
                        #current_eye_ratio = (np.median(area_record)-eye_area)/np.median(area_record)
                        self.eyestate = 0 # 0 = blink
                        if self.ratio > self.blink_threshold.value(): #close eye
                            self.blink_counter +=1
                            self.colorBackgroundText(frame,  f'Blink', FONT, self.FONT_SIZE, (int(frame_height/2), 100), 2, (0,255,255), pad_x=6, pad_y=6, )
                        else: #open eye
                            if self.blink_counter >= self.eye_close_frame :
                                self.eyestate = 1 # 1 = blink
                                self.total_blink +=1
                                self.blink_counter =0
                        if (self.eye_area_record/self.eye_area)**0.5 < self.distance_threshold.value():
                            self.area_counter +=1
                            if(self.area_counter>2):
                                self.colorBackgroundText(frame,  f'Too close', FONT, self.FONT_SIZE, (int(frame_height/2), 150), 2, (0,255,255), pad_x=6, pad_y=6, )
                                if(self.area_counter>60):
                                    self.showDialog('Too close',line=True)
                        else:
                            self.area_counter = 0

                        if self.brightness_value <self.bright_threshold.value():
                            self.bright_counter +=1
                            if(self.bright_counter >20):
                                self.colorBackgroundText(frame,  f'Too dim', FONT, self.FONT_SIZE, (int(frame_height/2), 150), 2, (0,255,255), pad_x=6, pad_y=6, )
                                #self.showDialog('Too close')
                        else:
                            self.bright_counter = 0

                        # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                        self.colorBackgroundText(frame,  f'Total Blinks: {self.total_blink}', FONT, self.FONT_SIZE/2, (30,150),2)
                        cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.LEFT_EYE ], dtype=np.int32)], True,(0,255,0), 1, cv.LINE_AA)
                        cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv.LINE_AA)
                        cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.FACE_OVAL ], dtype=np.int32)], True, (0,0,255), 1, cv.LINE_AA)
                        #cv.polylines(frame,  [np.array([mesh_coords[p] for p in LIPS ], dtype=np.int32)], True, utils.RED, 1, cv.LINE_AA)
                        #cv.polylines(frame,  [np.array([mesh_coords[p] for p in CENTER_POINT ], dtype=np.int32)], True, utils.RED, 1, cv.LINE_AA)
                    else:
                        self.previous_time_step =  time.time()
                        self.record_state = 1 # do not detect people
                #print(PolyArea(np.array([mesh_coords[p] for p in LEFT_EYE ])[:,0],np.array([mesh_coords[p] for p in LEFT_EYE ])[:,1]))
                # calculating  frame per seconds FPS
                self.fps_pass_time = time.time()-self.start_time
                fps = self.frame_counter/self.fps_pass_time
                self.colorBackgroundText(frame,  f'Eye area : {(self.eye_area)}', FONT, self.FONT_SIZE/2, (30,90),1)
                self.colorBackgroundText(frame,  f'Eye Distance ratio: {round((self.eye_area_record/self.eye_area)**0.5,2)}', FONT, self.FONT_SIZE/2, (30,120),1)
                self.colorBackgroundText(frame,  f'Eye Ratio: {round(self.ratio,2)}', FONT, self.FONT_SIZE/2, (30,150),1)
                self.colorBackgroundText(frame,  f'Brightness: {round(self.brightness_value,1)}', FONT, self.FONT_SIZE/2, (30,180),1)
                #self.colorBackgroundText(frame,  f'Brightness: {round(brightness,1)}', FONT, 0.7, (30,300),2)
                self.colorBackgroundText(frame,  f'FPS: {round(fps,1)}', FONT,self.FONT_SIZE/2, (30,60),1)
                # writing image for thumbnail drawing shape
                # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
                show = cv.resize(frame,(800,600))
                show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(showImage))

        
            elif(self.exercise.currentText() == 'jumping jack'):
                FONTS =cv.FONT_HERSHEY_COMPLEX
                self.record_state = 1
                with self.mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,model_complexity=0) as pose:
                    success, image = self.camera.read()
                    image = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
                    # To improve performance, optionally mark the image as not writeable to
                    image.flags.writeable = False
                    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                    results = pose.process(image)
                    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
                    if results.pose_world_landmarks:
                        self.record_state = 0
                        mesh_coords = self.landmarksDetection(image, results, False,True)
                        if (self.get_state_body(results)  == -self.previous_state):
                            self.previous_state = self.get_state_body(results)
                            self.count_hand += 1
                            print(self.count_hand,self.count_jump)

                        self.count = self.count_hand
                        image.flags.writeable = True
                        self.mp_drawing.draw_landmarks(
                            image,
                            results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS)
                        #cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.LEFT_EYE ], dtype=np.int32)], True,(0,255,0), 1, cv.LINE_AA)
                    image = cv.flip(image, 1)
                    image = self.colorBackgroundText(image,  f'Total : {int(self.count/2)}', FONTS, 0.7, (30,200),1)        
                    # Flip the image horizontally for a selfie-view display.
                    #cv2.imshow('MediaPipe Pose', image)
                    show = cv.resize(image,(800,600))
                    show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
                    showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
                    self.camera_label.setPixmap(QPixmap.fromImage(showImage))

            elif(self.exercise.currentText() == 'close eye' or self.exercise.currentText() == 'None'):
                with self.map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
                    self.frame_counter += 1
                    ret, frame = self.camera.read() 
                    frame_height, frame_width= frame.shape[:2]
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    results  = face_mesh.process(rgb_frame)
                    FONT = cv.FONT_HERSHEY_COMPLEX
                    if results.multi_face_landmarks:
                        self.record_state = 0
                        mesh_coords = self.landmarksDetection(frame, results, False)
                        right_eye_area = self.PolyArea(np.array([mesh_coords[p] for p in self.RIGHT_EYE ])[:,0],np.array([mesh_coords[p] for p in self.RIGHT_EYE ])[:,1])
                        left_eye_area = self.PolyArea(np.array([mesh_coords[p] for p in self.LEFT_EYE ])[:,0],np.array([mesh_coords[p] for p in self.LEFT_EYE ])[:,1])
                        self.eye_area = (right_eye_area+left_eye_area)/2
                        self.ratio = self.blinkRatio(frame, mesh_coords, self.RIGHT_EYE, self.LEFT_EYE)
                        self.brightness_value = self.get_average_brightness(rgb_frame,mesh_coords,frame_height,frame_width) 

                        if (self.ratio > self.blink_threshold.value() or self.exercise.currentText() == 'None'): #close eye
                            self.eyestate = 1 # 1 = blink
                            self.pass_time += (time.time() - self.previous_time_step)
                            self.previous_time_step =  time.time()
                            self.colorBackgroundText(frame,  f'Close', FONT, self.FONT_SIZE, (int(frame_height/2), 100), 2, (0,255,255), pad_x=6, pad_y=6, )
                        else: #open eye
                            self.eyestate = 0 # 0 = blink
                            self.pass_time += 0
                            self.previous_time_step = time.time()
    
                        cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.LEFT_EYE ], dtype=np.int32)], True,(0,255,0), 1, cv.LINE_AA)
                        cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv.LINE_AA)
                        cv.polylines(frame,  [np.array([mesh_coords[p] for p in self.FACE_OVAL ], dtype=np.int32)], True, (0,0,255), 1, cv.LINE_AA)
              #print(PolyArea(np.array([mesh_coords[p] for p in LEFT_EYE ])[:,0],np.array([mesh_coords[p] for p in LEFT_EYE ])[:,1]))
                # calculating  frame per seconds FPS
                self.fps_pass_time = time.time()-self.start_time
                fps = self.frame_counter/self.fps_pass_time
                self.colorBackgroundText(frame,  f'Eye area : {(self.eye_area)}', FONT, self.FONT_SIZE/2, (30,90),1)
                self.colorBackgroundText(frame,  f'Eye Distance ratio: {(self.eye_area_record/self.eye_area)**0.5}', FONT, self.FONT_SIZE/3, (30,120),1)
                self.colorBackgroundText(frame,  f'Eye Ratio: {round(self.ratio,3)}', FONT, self.FONT_SIZE/2, (30,150),1)
                self.colorBackgroundText(frame,  f'Brightness: {round(self.brightness_value,1)}', FONT, self.FONT_SIZE/2, (30,180),1)
                #self.colorBackgroundText(frame,  f'Brightness: {round(brightness,1)}', FONT, self.FONT_SIZE/3, (30,300),2)
                self.colorBackgroundText(frame,  f'FPS: {round(fps,1)}', FONT, self.FONT_SIZE/2, (30,60),1)
                # writing image for thumbnail drawing shape
                # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
                show = cv.resize(frame,(800,600))
                show = cv.cvtColor(show, cv.COLOR_BGR2RGB)
                showImage = QImage(show.data, show.shape[1],show.shape[0],QImage.Format_RGB888)
                self.camera_label.setPixmap(QPixmap.fromImage(showImage))
                
            elif(self.exercise.currentText() == 'None'):
                self.pass_time = (time.time() - self.previous_time_step)
                self.time_status = 'relax'

            if (self.status == 'start' or self.status== 'rest' ):
                if(self.status == 'start'):
                    remain_time =  self.work_time*60 - self.pass_time # self.work_time*60 - ( time.time() - self.init_time)
                elif(self.status == 'rest'):
                    remain_time =  self.rest_time*60 - self.pass_time # self.work_time*60 - ( time.time() - self.init_time)
                #print(remain_time,type(remain_time),float(remain_time))
                hour = remain_time // 3600
                minute = (remain_time - (hour * 3600)) // 60
                second = (remain_time - (hour * 3600) - (minute * 60))
                #print("Total time:", time, "sec")
                #print("Time remain:", hour, "hr", minute, "min", second, "sec")
                self.Progress_progressBar.setValue(int(self.pass_time/(remain_time+self.pass_time)*100))
                self.Time_Hour_lcdNumber.display(str(int(hour)))
                self.Time_Minute_lcdNumber.display(str(int(minute)))
                self.Time_Second_lcdNumber.display(str(int(second)))
                self.count_minute += 1 
                self.count_bright += self.brightness_value
                self.count_blink += self.eyestate
                self.count_distance += (self.eye_area_record/self.eye_area)**0.5
                pass_minute = ( time.time() - self.init_time) // 60
                if (pass_minute > self.previous_minute):
                    print('save')
                    self.previous_minute = pass_minute
                    bright_avg = int(self.count_bright/self.count_minute)
                    blink_avg = self.count_blink
                    distance_avg = round(self.count_distance/self.count_minute,3)
                    self.count_bright = 0
                    self.count_blink = 0
                    self.count_distance = 0
                    self.count_minute = 0
                    result = time.localtime(time.time())
                    if(self.status == 'start'):
                        self.cursorObj.execute("insert or ignore into %s(year, month, day, hour, minute, distance, brightness, blink,state) VALUES (?,?,?,?,?,?,?,?,?)" %self.current_user, \
                        (int(result.tm_year), int(result.tm_mon), int(result.tm_mday), int(result.tm_hour), int(result.tm_min), distance_avg, bright_avg, blink_avg,self.record_state))
                    elif(self.status == 'rest'):
                        self.cursorObj.execute("insert or ignore into %s(year, month, day, hour, minute, distance, brightness, blink,state) VALUES (?,?,?,?,?,?,?,?,?)" %self.current_user, \
                        (int(result.tm_year), int(result.tm_mon), int(result.tm_mday), int(result.tm_hour), int(result.tm_min), 1, 10, 10,self.record_state))
                    self.con.commit() 
            
                if (remain_time<0 and self.status=='start'):
                    print('rest')
                    self.status = 'rest'
                    self.pass_time = 0.01
                    self.previous_time_step = time.time()
                    self.blink_counter = 0
                    self.showDialog('Rest Now',line=True)

                elif((remain_time<0  or self.count >= self.excerise_count.value()) and self.status=='rest'):
                    print('finish rest')
                    self.showDialog('finish rest')
                    self.count = 0
                    self.count_hand = 0
                    self.status = 'start'
                    self.pass_time = 0.01
                    self.blink_counter = 0
                    self.start_push_onchange()
        except Exception as e: 
            print(e)
            pass
        

if __name__ == '__main__':
    app = QApplication([])
    #apply_stylesheet(app, theme='dark_blue.xml')
    window = Window()
    window.show()
    app.exec()
