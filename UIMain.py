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
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 30, 631, 691))
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.widget = QtWidgets.QWidget(self.tab)
        self.widget.setGeometry(QtCore.QRect(440, 70, 138, 219))
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.Button_StopTrain = QtWidgets.QPushButton(self.widget)
        self.Button_StopTrain.setObjectName("Button_StopTrain")
        self.verticalLayout_3.addWidget(self.Button_StopTrain)
        self.Button_SaveModel = QtWidgets.QPushButton(self.widget)
        self.Button_SaveModel.setObjectName("Button_SaveModel")
        self.verticalLayout_3.addWidget(self.Button_SaveModel)
        self.Button_LoadModel = QtWidgets.QPushButton(self.widget)
        self.Button_LoadModel.setObjectName("Button_LoadModel")
        self.verticalLayout_3.addWidget(self.Button_LoadModel)
        self.Button_ReadModel = QtWidgets.QPushButton(self.widget)
        self.Button_ReadModel.setObjectName("Button_ReadModel")
        self.verticalLayout_3.addWidget(self.Button_ReadModel)
        self.Button_Train = QtWidgets.QPushButton(self.widget)
        self.Button_Train.setObjectName("Button_Train")
        self.verticalLayout_3.addWidget(self.Button_Train)
        self.widget1 = QtWidgets.QWidget(self.tab)
        self.widget1.setGeometry(QtCore.QRect(0, 0, 431, 661))
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
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
        self.Button_StopTrain.setText(_translate("MainWindow", "训练强制停止"))
        self.Button_SaveModel.setText(_translate("MainWindow", "模型保存"))
        self.Button_LoadModel.setText(_translate("MainWindow", "网络模型导入"))
        self.Button_ReadModel.setText(_translate("MainWindow", "模型读取"))
        self.Button_Train.setText(_translate("MainWindow", "神经网络训练"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "模型训练"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "模型识别"))
        self.dockWidget.setWindowTitle(_translate("MainWindow", "Log栏"))
