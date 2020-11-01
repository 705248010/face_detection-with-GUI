# -*- coding: utf-8 -*-

from ui.infoUI import *


class InfoUI(QtWidgets.QMainWindow, Ui_Form):
    def __init__(self):
        super(InfoUI, self).__init__()
        self.setupUi(self)
