# -*- coding: utf-8 -*-

from ui.emailUI import *
from PyQt5.QtWidgets import QMessageBox
from log import Logging
import os


class EmailUI(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(EmailUI, self).__init__()
        self.setupUi(self)
        self.logger = Logging()
        self.read_email_conf()

    # 保存邮箱配置
    def save_email_conf(self):
        receiver_email = self.lineEdit.text()
        sender = self.lineEdit_2.text()
        receiver = self.lineEdit_3.text()
        if not (receiver_email and sender and receiver):
            QMessageBox.warning(self, '警告', '有字段为空')
            self.logger.log_error('email info save filed')
            return
        conf = {
            'receiver_email': receiver_email,
            'sender': sender,
            'receiver': receiver
        }
        with open('email_conf/conf.yaml', 'w', encoding='utf-8') as f:
            f.write(str(conf))
        QMessageBox.about(self, 'info', '保存成功')
        self.logger.log_info('email info save success')

    # 读取邮箱配置
    def read_email_conf(self):
        if os.path.exists('email_conf/conf.yaml'):
            with open('email_conf/conf.yaml', 'r', encoding='utf-8') as f:
                conf = eval(f.read())
            if conf:
                self.lineEdit.setText(conf['receiver_email'])
                self.lineEdit_2.setText(conf['sender'])
                self.lineEdit_3.setText(conf['receiver'])
            else:
                return
        else:
            return
