import gzip
import os
import sys
from datetime import datetime
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow
from DeepLearning import DeepLearning
from UIMain import Ui_MainWindow
from qt_material import apply_stylesheet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
import numpy as np


class MyClass(QMainWindow, Ui_MainWindow):
    Ver = "版本：1.0.0.7       更新日期：20230407"

    # 深度学习类实例化
    dl = DeepLearning()
    # 损失函数函数图像X坐标
    xloss = []
    # 损失函数函数图像Y坐标
    yloss = []
    # 识别率函数图像X坐标
    xacc = []
    # 识别率函数图像Y坐标
    yacc = []

    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent=parent)
        self.init()
        # 绑定深度学习类中textWritten信号，此信号来刷新Log
        self.dl.textWritten.connect(self.write_text)
        # 绘图事件绑定
        self.dl.drawMat.connect(self.DrawMat)
        # 刷新底部状态栏显示相关信息
        self.statusBar().showMessage(self.Ver)

    # 初始化控件&&绑定控件事件
    def init(self):
        self.setupUi(self)
        self.label.setStyleSheet("background-color:black")
        self.write_text(' 软件初始化成功')
        self.setWindowTitle('基于特征识别的手写数字识别')
        self.setWindowIcon(QIcon('./logo.ico'))

        # 构建Matplolib，并添加到控件
        self.fig = plt.Figure()
        self.canvas = FC(self.fig)
        self.verticalLayout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)

        # 训练开始按钮绑定
        self.Button_Train.clicked.connect(self.run)
        self.Button_LoadModel.clicked.connect(self.DrawMat)

        self.pushButton.clicked.connect(self.ShowImage)

    # 这里是对深度学习训练运行的调用
    def run(self):
        # 初始化图标
        self.ax.cla()
        self.ax.set(xlim=[0, 12000], ylim=[0, 1], title='Loss',
                    ylabel='Loss && Acc', xlabel='Number of trainings')
        self.xloss.clear()
        self.yloss.clear()

        # 训练开始运行
        self.dl.start()

    # 绘图事件
    def DrawMat(self, xvalue, yvalue, xacc, yacc):
        self.xloss.append(xvalue)
        self.yloss.append(yvalue)
        self.xacc.append(xacc)
        self.yacc.append(yacc)
        self.ax.plot(self.xloss, self.yloss, linewidth='1', label="Loss", color='orangered')
        self.ax.plot(self.xacc, self.yacc, linewidth='1', label="Acc", color='springgreen')
        self.canvas.draw()

    # Log记录
    def write_text(self, str):
        now = datetime.now()
        Mes = now.strftime("%H:%M:%S") + str
        self.textEdit.append(Mes)

    # 显示图像按钮
    def ShowImage(self):
        files = ['train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz',
                 't10k-images-idx3-ubyte.gz']
        paths = []
        for fname in files:
            paths.append(os.path.join('MNIST/', fname))
        if len(paths) == 0:
            return '路径异常，请检测路径设置是否正确'

        with gzip.open(paths[0], 'rb') as lbpath:
            train_labels = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(paths[1], 'rb') as imgpath:
            train_images = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(train_labels), 28, 28)

        with gzip.open(paths[2], 'rb') as lbpath:
            test_label = np.frombuffer(lbpath.read(), np.uint8, offset=8)
        with gzip.open(paths[3], 'rb') as imgpath:
            test_image = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(test_label), 28, 28)

        image_index = 0
        image = train_images[image_index]

        qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_Grayscale8)
        scaled_image = QPixmap.fromImage(qimage).scaled(300, 300)
        self.label.setPixmap(scaled_image)


if __name__ == '__main__':
    ['dark_amber.xml',
     'dark_blue.xml',
     'dark_cyan.xml',
     'dark_lightgreen.xml',
     'dark_pink.xml',
     'dark_purple.xml',
     'dark_red.xml',
     'dark_teal.xml',
     'dark_yellow.xml',
     'light_amber.xml',
     'light_blue.xml',
     'light_cyan.xml',
     'light_cyan_500.xml',
     'light_lightgreen.xml',
     'light_pink.xml',
     'light_purple.xml',
     'light_red.xml',
     'light_teal.xml',
     'light_yellow.xml']
    app = QApplication(sys.argv)
    Mc = MyClass()
    apply_stylesheet(app, theme='light_pink.xml')
    Mc.show()
    sys.exit(app.exec_())
