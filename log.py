# -*- coding: utf-8 -*-

import logging


class Logging:
    def __init__(self):
        # 日志格式
        self.LOG_FORMAT = "%(asctime)s[%(levelname)s] - %(message)s"

        # 日志等级、输出、格式设置
        logging.basicConfig(filename='programs_logs/logging.log', level=logging.DEBUG, format=self.LOG_FORMAT)

    def log_info(self, text):
        logging.info(text)

    def log_warning(self, text):
        logging.warning(text)

    def log_error(self, text):
        logging.error(text)

    def log_critical(self, text):
        logging.critical(text)
