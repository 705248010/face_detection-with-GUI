# -*- coding: utf-8 -*-

from ui.dataManageUI import *
from log import Logging
from database.sqliteManage import SqlLiteManage
from PyQt5.QtWidgets import QMessageBox
from PIL import Image
import numpy as np
import cv2
import os


class DataManageUI(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(DataManageUI, self).__init__()
        self.setupUi(self)

        # sqlite
        self.sqlite = SqlLiteManage()

        # 日志
        self.logger = Logging()

        # 数据集路径
        self.dataset_path = './dataset'

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
        self.show_table()

    # 查询
    # 考虑字段是否存在
    def search_item(self):
        if not self.lineEdit.text():
            QMessageBox.warning(self, '警告', '查询字段为空，请重试')
            self.logger.log_warning('field is null')
            self.logger.log_error('search failed')
        else:
            field = self.comboBox.currentText()
            if field == '姓名':
                field = 'USR_NAME'
                value = self.lineEdit.text()
            else:
                field = 'FACE_ID'
                value = int(self.lineEdit.text())
            isCheacked, data = self.sqlite.search(field, value)
            if isCheacked:
                result = {
                    'FACE_ID': data[0],
                    'USR_NAME': data[1],
                    'USR_NAME_CHAR': data[2],
                    'IS_TRAINED': data[3]
                }
                self.textBrowser.clear()
                self.textBrowser.setText(str(result))
                self.logger.log_info('search data success, field is %s, data is %s' % (field, str(result)))
            else:
                QMessageBox.warning(self, '警告', '未查询到字段')
                self.logger.log_warning('field not exists')
                self.logger.log_error('search failed')
                return

    # 删除
    # 考虑字段是否存在
    def remove_item(self):
        if not self.lineEdit_2.text():
            QMessageBox.warning(self, '警告', '字段为空，请重试')
            self.logger.log_warning('field is null')
            self.logger.log_error('delete failed')
        else:
            field = self.comboBox_2.currentText()
            if field == '姓名':
                field = 'USR_NAME'
            else:
                field = 'FACE_ID'
            value = self.lineEdit_2.text()
            data = self.sqlite.remove(field, value)
            if data:
                result = {
                    'FACE_ID': data[0],
                    'USR_NAME': data[1],
                    'USR_NAME_CHAR': data[2],
                    'IS_TRAINED': data[3]
                }
                self.logger.log_info('delete data success, field is %s, data is %s' % (field, str(result)))
            else:
                QMessageBox.warning(self, '警告', '未查询到字段')
                self.logger.log_warning('field not exists')
                self.logger.log_error('delete failed')
                return
        self.show_table()

    # 插入数据
    def insert_data(self):
        if not (self.lineEdit_3.text() and self.lineEdit_4.text() and self.lineEdit_5.text()):
            QMessageBox.warning(self, '警告', '部分字段为空，请填写合适数据，否则这段数据不会被插入')
            self.logger.log_warning('some fields are null')
            self.logger.log_error('insert data failed')
        else:
            values = (int(self.lineEdit_3.text()), self.lineEdit_4.text(), self.lineEdit_5.text(), 0)
            try:
                self.sqlite.insert(values)
                self.logger.log_info('insert data success')
                self.show_table()
            except:
                QMessageBox.warning(self, '警告', '数据库中已经有ID相同字段，请重试')
                self.logger.log_warning('already have the same data')
                self.logger.log_error('insert data failed')

    # 训练
    def train(self):
        if not os.listdir('./dataset'):
            QMessageBox.warning(self, '警告', '数据集为空，请先采集人脸')
            self.logger.log_warning('dataset is null')
            return
        if self.sqlite.iter_search()[-1][-1] == 0:
            self.train_model()
        else:
            if self.sqlite.iter_search()[-1][-1] == 1:
                isSelected = QMessageBox.question(self, '确认', '检测到所有人脸都已经训练过，是否继续？',
                                                  QMessageBox.Ok | QMessageBox.Cancel,
                                                  QMessageBox.Ok)
                if isSelected == QMessageBox.Ok:
                    self.train_model()
                elif isSelected == QMessageBox.Cancel:
                    self.logger.log_info('usr not select "train face data"')
                else:
                    return

    # 训练模型
    def train_model(self):
        self.logger.log_info('train start')
        # LBPH人脸识别器
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        # 级联分类器，检测人脸
        detector = cv2.CascadeClassifier(
            "E:\Opencv\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")

        def get_face_and_id():
            path = './dataset'
            images_path = [os.path.join(path, f) for f in os.listdir(path)]
            face_samples = []
            face_ids = []
            for imagePath in images_path:
                # 读取图片
                PIL_img = Image.open(imagePath).convert('L')
                # 以uint8矩阵存储
                img_numpy = np.array(PIL_img, 'uint8')
                # 提取id
                face_id = int(os.path.split(imagePath)[-1].split(".")[1])
                # 第一个参数img
                # 第二个参数scale_factor，数值越大计算越快，太快会丢失人脸
                # 第三个参数minNeighbors，控制着误检测，默认值为3。表明至少有3次重叠检测，我们才认为人脸确实存在
                faces = detector.detectMultiScale(img_numpy, 1.1, 3)
                # 提取ROI
                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y + h, x:x + w])
                    face_ids.append(face_id)
            return face_samples, face_ids

        # 获取脸部和id
        faces, ids = get_face_and_id()
        # 训练识别器
        recognizer.train(faces, np.array(ids))

        # 将训练后的模型存入./trainer/trainer.yml
        recognizer.write('./trainer/trainer.yml')

        # 修改数据库字段
        self.sqlite.change_train_field()
        self.show_table()

        # 打印训练模型脸的数量
        self.logger.log_info("{} faces trained".format(len(np.unique(ids))))
        self.logger.log_info('train end')

    # tableWidgets显示表格
    # 初始化表格
    def show_table(self):
        rows_number = self.sqlite.count_data()
        # 设置行列
        self.tableWidget.setRowCount(rows_number)
        self.tableWidget.setColumnCount(4)
        # 设置表头
        self.tableWidget.setHorizontalHeaderLabels(['人脸ID', '姓名', '姓名拼音', '是否训练过'])
        # 设置表格自适应
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # 设置表格只读
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 获取数据库中所有数据，并格式化显示
        # 为每行每列添加数据
        data = self.sqlite.iter_search()
        row = 0
        for item in data:
            column = 0
            for i in range(4):
                newItem = QtWidgets.QTableWidgetItem(str(item[i]))
                self.tableWidget.setItem(row, column, newItem)
                column += 1
            row += 1
