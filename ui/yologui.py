# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(964, 730)
        font = QtGui.QFont()
        font.setPointSize(12)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Loadmodelbt = QtWidgets.QPushButton(self.centralwidget)
        self.Loadmodelbt.setGeometry(QtCore.QRect(820, 10, 131, 31))
        self.Loadmodelbt.setObjectName("Loadmodelbt")
        self.resultdisplay = QtWidgets.QLabel(self.centralwidget)
        self.resultdisplay.setGeometry(QtCore.QRect(20, 90, 931, 521))
        self.resultdisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.resultdisplay.setObjectName("resultdisplay")
        self.dir_show = QtWidgets.QLineEdit(self.centralwidget)
        self.dir_show.setGeometry(QtCore.QRect(10, 10, 801, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.dir_show.setFont(font)
        self.dir_show.setText("")
        self.dir_show.setObjectName("dir_show")
        self.Loadvideobt = QtWidgets.QPushButton(self.centralwidget)
        self.Loadvideobt.setGeometry(QtCore.QRect(820, 50, 131, 31))
        self.Loadvideobt.setObjectName("Loadvideobt")
        self.dir_video_show = QtWidgets.QLineEdit(self.centralwidget)
        self.dir_video_show.setGeometry(QtCore.QRect(10, 50, 801, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.dir_video_show.setFont(font)
        self.dir_video_show.setText("")
        self.dir_video_show.setObjectName("dir_video_show")
        self.m_display = QtWidgets.QLCDNumber(self.centralwidget)
        self.m_display.setGeometry(QtCore.QRect(90, 630, 81, 41))
        self.m_display.setObjectName("m_display")
        self.s_display = QtWidgets.QLCDNumber(self.centralwidget)
        self.s_display.setGeometry(QtCore.QRect(260, 630, 81, 41))
        self.s_display.setObjectName("s_display")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 630, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(190, 630, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.startbt = QtWidgets.QPushButton(self.centralwidget)
        self.startbt.setGeometry(QtCore.QRect(390, 630, 111, 41))
        self.startbt.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.startbt.setObjectName("startbt")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 964, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Loadmodelbt.setText(_translate("MainWindow", "Selected model"))
        self.resultdisplay.setText(_translate("MainWindow", "Result Display"))
        self.Loadvideobt.setText(_translate("MainWindow", "Selected Video"))
        self.label.setText(_translate("MainWindow", "Main"))
        self.label_2.setText(_translate("MainWindow", "Sub"))
        self.startbt.setText(_translate("MainWindow", "Start"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())