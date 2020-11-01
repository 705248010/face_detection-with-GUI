# -*- coding: utf-8 -*-

import smtplib
from log import Logging
from email.mime.text import MIMEText
from email.header import Header


class Mail:
    def __init__(self):
        self.mail_host = 'smtp.qq.com'
        self.mail_pass = ''  # 授权码
        self.sender = ''  # 发送人
        self.receiver, self.sender_name, self.receiver_name = self.read_conf()
        self.logger = Logging()

    # 读取配置
    def read_conf(self):
        with open('E:/Opencv/FaceRecognition_demo/email_conf/conf.yaml', 'r', encoding='utf-8') as f:
            conf = eval(f.read())
        return [conf['receiver_email']], conf['sender'], conf['receiver']

    # 发邮件
    def send(self, text):
        # 邮件内容
        content = text
        message = MIMEText(content, 'plain', 'utf-8')

        # 寄件人和收件人
        message['From'] = Header(self.sender_name, 'utf-8')
        message['To'] = Header(self.receiver_name, 'utf-8')

        subject = 'python_test'  # 发送的主题，可自由填写
        message['Subject'] = Header(subject, 'utf-8')

        try:
            smtpObj = smtplib.SMTP_SSL(self.mail_host, 465)
            smtpObj.login(self.sender, self.mail_pass)
            smtpObj.sendmail(self.sender, self.receiver, message.as_string())
            smtpObj.quit()
            self.logger.log_info('send mail success')
        except smtplib.SMTPException as e:
            self.logger.log_error('send mail failed')
