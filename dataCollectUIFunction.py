# -*- coding: utf-8 -*-

from ui.dataCollectUI import *
from log import Logging
from database.sqliteManage import SqlLiteManage
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtGui import QPixmap, QImage
import time
import cv2
import os


class DataCollectUI(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(DataCollectUI, self).__init__()
        self.setupUi(self)

        # sqlite
        self.sqlite = SqlLiteManage()

        # 日志
        self.logger = Logging()

        # 摄像头
        self.capture = cv2.VideoCapture()

        # 定时器
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # 当前人脸ID
        self.faceID = 0

        # 是否进行人脸采集
        self.isCollect = False
        self.isContinueCollect = False

        # 是否打开摄像头
        self.isOpenCapture = False

        # 自定义参数
        self.count = 0  # 人脸采集计数
        self.MaxCount = 50  # 采集人脸数量
        self.scale_factor = 1.1  # 数值越大计算越快，太快会丢失人脸
        self.minNeighbors = 3  # 控制着误检测

        # 预处理选择
        self.isGaussianBlur = False  # 高斯
        self.isBlur = False  # 均值
        self.isMedianBlur = False  # 中值
        self.isEqualizeHist = False  # 直方图均衡

        # 级联分类器
        self.face_detector = cv2.CascadeClassifier(
            "E:\Opencv\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")

        # 人脸数量
        self.max_face_count = 1

    # 初始化
    def root(self):
        isSelected = QMessageBox.question(self, '确认', '初始化会清空表中所有数据，确定要这么做吗？', QMessageBox.Ok | QMessageBox.Cancel,
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

    # 打开摄像头
    # 管理外接摄像头
    def open_capture(self):
        self.logger.log_info('open capture')
        self.isOpenCapture = True
        capture_dev_id = 0
        if self.checkBox.isChecked():
            capture_dev_id = 1
        self.capture.open(capture_dev_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 721)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 511)
        ret, frame = self.capture.read()
        self.timer.start(5)

    # 关闭摄像头
    def close_capture(self):
        self.logger.log_info('close capture')
        self.capture.release()
        self.timer.stop()
        self.label_3.setText("摄像头识别区")
        self.isOpenCapture = False

    # 人脸检测
    def face_detection(self, frame):
        min_width = 0.1 * self.capture.get(3)
        min_height = 0.1 * self.capture.get(4)

        # 灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=3,
            minSize=(int(min_width), int(min_height)),
        )

        # 有多张人脸时暂停采集，一张人脸继续采集
        if len(faces) > self.max_face_count:
            self.isContinueCollect = False
        else:
            self.isContinueCollect = True

        # 用矩形画出脸的位置
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame

    # 人脸采集
    # 录入后，只给最后一条ID录人脸
    def face_collect(self):
        if not self.isOpenCapture:
            QMessageBox.warning(self, '警告', '未打开摄像头')
            self.logger.log_warning('capture not open')
            return
        if self.sqlite.count_data() == 0:
            QMessageBox.warning(self, '警告', '数据库为空')
            self.logger.log_warning('database is null')
            return
        isSelected = QMessageBox.question(self, '确认', '人脸采集只会给最新录入的数据采集，是否继续？', QMessageBox.Ok | QMessageBox.Cancel,
                                          QMessageBox.Ok)
        if isSelected == QMessageBox.Ok:
            self.logger.log_info('start collect face...')

            # 自定义初始化
            self.MaxCount = int(self.label_8.text())
            self.scale_factor = float(self.doubleSpinBox.value())
            self.minNeighbors = int(self.label_10.text())
            self.isGaussianBlur = self.radioButton.isChecked()
            self.isBlur = self.radioButton_2.isChecked()
            self.isMedianBlur = self.radioButton_3.isChecked()
            self.isEqualizeHist = self.checkBox_5.isChecked()

            self.count = 0
            self.faceID = self.sqlite.iter_search()[-1][0]
            self.isCollect = True
        elif isSelected == QMessageBox.Cancel:
            self.logger.log_info('usr not select "face_collect"')
        else:
            return

    # 定时刷新界面
    def update_frame(self):
        start = time.time()
        ret, frame = self.capture.read()
        img = self.face_detection(frame)
        if ret and self.isCollect and self.isContinueCollect:
            end = time.time()
            img = cv2.putText(img, 'FPS: %.1f' % (1 / (end - start)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 255, 255), 1)
            img = cv2.putText(img, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), (210, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            self.frame_dispose(img)
            self.collect_face_data(frame)
        else:
            end = time.time()
            img = cv2.putText(img, 'FPS: %.1f' % (1 / (end - start)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                              (255, 255, 255), 1)
            img = cv2.putText(img, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), (210, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            self.frame_dispose(img)
        if self.count >= self.MaxCount:
            self.isCollect = False
            self.logger.log_info('face data collect finish')
            self.close_capture()

    # 人脸采集
    def collect_face_data(self, frame):
        # 灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 三种滤波只能选择一种
        if self.isGaussianBlur:
            gray = cv2.GaussianBlur(gray, (5, 5), 1)
        if self.isMedianBlur:
            gray = cv2.medianBlur(gray, 5)
        if self.isBlur:
            gray = cv2.blur(gray, (3, 3))

        # 直方图均衡
        if self.isEqualizeHist:
            gray = cv2.equalizeHist(gray)

        # 第一个参数img
        # 第二个参数scale_factor，数值越大计算越快，太快会丢失人脸
        # 第三个参数minNeighbors，控制着误检测，默认值为3。表明至少有3次重叠检测，我们才认为人脸确实存在
        faces = self.face_detector.detectMultiScale(gray, self.scale_factor, self.minNeighbors)

        # 图像中人脸大于1就跳过
        if len(faces) > self.max_face_count:
            pass

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.count += 1
            self.lcdNumber.display(self.count)

            # 将照片存入指定文件夹./dataset
            cv2.imwrite("./dataset/User." + str(self.faceID) + '.' + str(self.count) + ".jpg", gray[y:y + h, x:x + w])

    # 对每一帧进行处理
    def frame_dispose(self, frame):
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
        self.label_3.setPixmap(QPixmap.fromImage((output_img)))
        self.label_3.setScaledContents(True)

    # 添加新数据
    def add_item(self):
        if not (self.lineEdit.text() and self.lineEdit_2.text() and self.lineEdit_3.text()):
            QMessageBox.warning(self, '警告', '部分字段为空，请填写合适数据，否则这段数据不会被插入')
            self.logger.log_warning('some fields are null')
            self.logger.log_error('insert data failed')
        else:
            values = (int(self.lineEdit_2.text()), self.lineEdit.text(), self.lineEdit_3.text(), 0)
            try:
                self.sqlite.insert(values)
                self.log(1, 'insert data success')
            except:
                QMessageBox.warning(self, '警告', '数据库中已经有ID相同字段，请重试')
                self.logger.log_warning('already have the same data')
                self.logger.log_error('insert data failed')
