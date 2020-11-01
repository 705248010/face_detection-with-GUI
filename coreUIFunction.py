# -*- coding: utf-8 -*-
from ui.coreUI import *
from dataCollectUIFunction import DataCollectUI
from dataManageUIFunction import DataManageUI
from infoUIFuncition import InfoUI
from emailFuction import EmailUI
from database.sqliteManage import SqlLiteManage
from schedule_config.schedule_func import Scheduler
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox
from imutils.object_detection import non_max_suppression
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from log import Logging
import datetime
import sys
import cv2
import time
import os
import re


class MyWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)

        # 初始化子窗口
        self.data_manage_ui = DataManageUI()
        self.data_collect_ui = DataCollectUI()
        self.info_ui = InfoUI()
        self.email_ui = EmailUI()

        # 摄像头
        self.capture = cv2.VideoCapture()

        # 定时器
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 日志
        self.logger = Logging()

        # sqlite
        self.sqlite = SqlLiteManage()

        # LBPH人脸识别器
        self.recognizer = ''

        # 行人检测
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        # 级联分类器
        self.cascadePath = ''
        self.faceCascade = ''

        # 人脸信息
        self.face_info = ''

        # 识别参数
        self.scaleFactor = 1.1
        self.minNeighbors = 3

        # 定时器
        self.scheduler = Scheduler()

        # 是否打开摄像头
        self.isOpenCapture = False

        # 是否人脸识别
        self.isFaceRecognition = False

        # 是否行人检测
        self.isPeopleDetection = False

        # 训练文件是否存在
        self.isTrainerExists = False

        # 线程池
        self.threadPool = ThreadPoolExecutor(os.cpu_count())

        # face_table
        self.face_dict = {}

        # name_table
        self.name_list = []

    # 打开摄像头
    def open_capture(self):
        self.logger.log_info('open capture')
        self.isOpenCapture = True
        capture_dev_id = 0
        if self.checkBox.isChecked():
            capture_dev_id = 1
        self.capture.open(capture_dev_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 651)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 471)

        # 初始化定时器、线程池，开始多线程运行
        self.scheduler.__init__()

        try:
            self.root_thread_pool()
            self.threadPool.submit(self.scheduler.run_scheduler)
            self.threadPool.submit(self.delete_logs)
            self.threadPool.submit(self.delete_repeat_log)
            self.logger.log_info('thread pool start')
        except:
            self.logger.log_error('thread pool start failed')
            return

        # 获取当前已经训练过的人脸usr
        self.get_name_dict()
        self.get_face_dict()

        self.timer.start(5)

    # 关闭摄像头
    def close_capture(self):
        self.isFaceRecognition = False
        self.isTrainerExists = False
        self.threadPool.shutdown(wait=False)
        self.scheduler.cancel_job()
        self.logger.log_info('close capture')
        self.capture.release()
        self.timer.stop()
        self.label.setText("摄像头识别区")

    # 退出
    def exit_app(self):
        self.logger.log_info('exit process')
        sys.exit()

    # 人脸采集
    def data_collect(self):
        self.data_collect_ui.show()

    # 数据采集
    def data_manage(self):
        self.data_manage_ui.show_table()
        self.data_manage_ui.show()

    # 信息
    def info(self):
        self.info_ui.show()

    # 初始化数据集、训练集
    def root(self):
        isSelected = QMessageBox.question(self, '确认', '初始化会删除所有数据，确定要这么做吗？', QMessageBox.Ok | QMessageBox.Cancel,
                                          QMessageBox.Ok)
        if isSelected == QMessageBox.Ok:
            self.sqlite.create_table()
            self.sqlite.root_db()
            for file in os.listdir('./dataset'):
                os.remove('./dataset/' + file)
            for file in os.listdir('./trainer'):
                os.remove('./trainer/' + file)
            self.logger.log_info('root database success')
        elif isSelected == QMessageBox.Cancel:
            self.logger.log_info('usr not select "root database"')
        else:
            return

    # 邮箱设置
    def email_conf(self):
        self.email_ui.show()

    # 定时刷新界面
    # 从这里识别
    def update_frame(self):
        start = time.time()
        ret, frame = self.capture.read()
        # 行人检测
        if self.isPeopleDetection:
            frame = self.people_detection(frame)
        # 人脸识别
        if self.isFaceRecognition and self.isTrainerExists:
            frame = self.face_recognition(frame)
        end = time.time()
        frame = cv2.putText(frame, 'FPS: %.1f' % (1 / (end - start)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 1)
        frame = cv2.putText(frame, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), (210, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        self.frame_dispose(frame)

    # 行人检测
    def people_detection(self, frame):
        rects, weights = self.hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.15)
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        # 用非极大抑制法去除重叠图像
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
        return frame

    # 开始人脸识别
    def open_face_recognition(self):
        if not self.isOpenCapture:
            QMessageBox.warning(self, '警告', '摄像头未打开')
            self.logger.log_warning('usr try to open face recognition, but capture not open')
            return
        self.root_face_recognition_conf()
        self.judge_trainer_exists()
        if not self.isTrainerExists:
            QMessageBox.warning(self, '警告', '训练模型不存在，请检查trainer文件夹')
            self.logger.log_warning('trainer/trainer.yml not exists')
            self.logger.log_error('open capture failed')
            return
        else:
            self.isFaceRecognition = True

    # 关闭人脸识别
    def close_face_recognition(self):
        self.isFaceRecognition = False

    # 打开行人检测
    def open_people_detection(self):
        if not self.isOpenCapture:
            QMessageBox.warning(self, '警告', '摄像头未打开')
            self.logger.log_warning('usr try to open people detection, but capture not open')
            return
        self.isPeopleDetection = True

    # 关闭行人检测
    def close_people_detection(self):
        self.isPeopleDetection = False

    # 人脸识别
    def face_recognition(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX

        face_id = 0
        usr_name = ''

        min_width = 0.1 * self.capture.get(3)
        min_height = 0.1 * self.capture.get(4)

        # 灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # gray 表示输入 grayscale 图像。
        # scaleFactor 表示每个图像缩减的比例大小。
        # minNeighbors 表示每个备选矩形框具备的邻近数量。数字越大，假正类越少。
        # minSize 表示人脸识别的最小矩形大小。
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=(int(min_width), int(min_height)),
        )

        # 识别器进行识别，并返回置信度
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            # 置信度越小越好
            if confidence < 100:
                for face in self.face_info:
                    if face_id == face[0]:
                        usr_name = face[1]
                    else:
                        pass
                confidence = "  {0}%".format(round(100 - confidence))
                if self.face_dict[usr_name]:
                    self.logger.log_info('{} entered lab'.format(usr_name))
                    self.set_face_dict(usr_name, False)
            else:
                usr_name = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                if self.face_dict[usr_name]:
                    self.logger.log_info('unknown entered lab')
                    self.set_face_dict(usr_name, False)
            cv2.putText(frame, str(usr_name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        return frame

    # 对每一帧进行处理
    def frame_dispose(self, frame):
        # 三通道转换
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:  # rows[0], cols[1], channels[2]
            if img.shape[2] == 4:
                # The image is stored using a 32-bit byte-ordered RGBA format (8-8-8-8)
                # A: alpha channel，不透明度参数。如果一个像素的alpha通道数值为0%，那它就是完全透明的
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        output_img = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        self.label.setPixmap(QPixmap.fromImage((output_img)))
        self.label.setScaledContents(True)

    # 初始化识别配置
    def root_face_recognition_conf(self):
        self.scaleFactor = float(self.doubleSpinBox.value())
        self.minNeighbors = int(self.label_5.text())

        # LBPH人脸识别器
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        # 级联分类器
        self.cascadePath = "E:\Opencv\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml"
        self.faceCascade = cv2.CascadeClassifier(self.cascadePath)

        # 人脸信息
        self.face_info = [face for face in self.sqlite.iter_search() if face[-1] != 0]

    # 检测训练文件是否存在
    def judge_trainer_exists(self):
        # 读取模型
        if not os.path.exists('./trainer/trainer.yml'):
            self.isTrainerExists = False
        self.recognizer.read('./trainer/trainer.yml')
        self.isTrainerExists = True

    # 初始化线程池
    def root_thread_pool(self):
        self.threadPool = ThreadPoolExecutor(os.cpu_count())

    # 删除日志
    def delete_logs(self):
        while True:
            dates = []
            with open('./programs_logs/logging.log', 'r') as f:
                logs_list = f.readlines()
            now = datetime.datetime.now()
            for i in range(0, 8):
                delta = datetime.timedelta(days=i)
                dates.append((now + delta).strftime("%m-%d"))
            for line in logs_list:
                if re.findall(r'(?<=2020-)\d{2}-\d{2}', line)[0] not in dates:
                    logs_list.remove(line)
                else:
                    pass
            with open('./programs_logs/logging.log', 'w') as f:
                result = ''
                for line in logs_list:
                    result += line
                f.write(result)
            time.sleep(1024)

    # 每3s运行一次来改变face_dict中的值
    def delete_repeat_log(self):
        while True:
            for name in self.face_dict.keys():
                self.face_dict[name] = True
            time.sleep(3)

    # 获取name_dict
    def get_name_dict(self):
        self.name_list = [name for name in self.sqlite.search('USR_NAME')]

    # 获取face_dict
    def get_face_dict(self):
        self.face_dict = {name: True for name in self.name_list}
        self.face_dict['unknown'] = True

    # 设置face_dict
    def set_face_dict(self, key, value):
        self.face_dict[key] = value


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mywindow = MyWindow()
    mywindow.show()
    sys.exit(app.exec_())
