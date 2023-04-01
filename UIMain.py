# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UIMain.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(929, 751)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(120, 110, 54, 12))
        self.label.setObjectName("label")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(30, 190, 531, 481))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.Button_Train = QtWidgets.QPushButton(self.tab)
        self.Button_Train.setGeometry(QtCore.QRect(50, 30, 151, 141))
        self.Button_Train.setObjectName("Button_Train")
        self.Button_LoadModel = QtWidgets.QPushButton(self.tab)
        self.Button_LoadModel.setGeometry(QtCore.QRect(220, 140, 200, 100))
        self.Button_LoadModel.setObjectName("Button_LoadModel")
        self.Button_SaveModel = QtWidgets.QPushButton(self.tab)
        self.Button_SaveModel.setGeometry(QtCore.QRect(300, 20, 200, 100))
        self.Button_SaveModel.setObjectName("Button_SaveModel")
        self.Button_StopTrain = QtWidgets.QPushButton(self.tab)
        self.Button_StopTrain.setGeometry(QtCore.QRect(40, 280, 200, 100))
        self.Button_StopTrain.setObjectName("Button_StopTrain")
        self.Button_ReadModel = QtWidgets.QPushButton(self.tab)
        self.Button_ReadModel.setGeometry(QtCore.QRect(270, 280, 200, 100))
        self.Button_ReadModel.setObjectName("Button_ReadModel")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.dockWidget.setObjectName("dockWidget")
        self.dockWidgetContents = QtWidgets.QWidget()
        self.dockWidgetContents.setObjectName("dockWidgetContents")
        self.gridLayout_26 = QtWidgets.QGridLayout(self.dockWidgetContents)
        self.gridLayout_26.setObjectName("gridLayout_26")
        self.textEdit = QtWidgets.QTextEdit(self.dockWidgetContents)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_26.addWidget(self.textEdit, 0, 0, 1, 1)
        self.dockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(1), self.dockWidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "基于特征处理的手写数字识别"))
        self.label.setText(_translate("MainWindow", "TextLabel"))
        self.Button_Train.setText(_translate("MainWindow", "神经网络训练"))
        self.Button_LoadModel.setText(_translate("MainWindow", "网络模型导入"))
        self.Button_SaveModel.setText(_translate("MainWindow", "模型保存"))
        self.Button_StopTrain.setText(_translate("MainWindow", "训练强制停止"))
        self.Button_ReadModel.setText(_translate("MainWindow", "模型读取"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "模型训练"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "模型识别"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "Log栏"))
