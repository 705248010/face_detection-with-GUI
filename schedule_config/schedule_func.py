# -*- coding: utf-8 -*-
from email_conf.email_conf import *
import schedule
import datetime


class Scheduler:
    def __init__(self):
        self.mail = Mail()

        # 是否取消功能
        self.isCancelSchedule = False

    # 定时要做的工作
    def job(self):
        content = ''
        self.mail.__init__()
        with open('E:/Opencv/FaceRecognition_demo/programs_logs/logging.log', 'r') as f:
            text = f.readlines()
        now = datetime.datetime.now().strftime("%m-%d")
        for line in text:
            if now in line:
                if "entered" in line:
                    content += line
                else:
                    pass
            else:
                pass
        self.mail.send(content)

    # 执行定时任务
    def run_scheduler(self):
        schedule.every(10).seconds.do(self.job)
        while True:
            schedule.run_pending()
            if self.isCancelSchedule:
                schedule.cancel_job(self.job)
                return

    # 取消任务
    def cancel_job(self):
        self.isCancelSchedule = True
