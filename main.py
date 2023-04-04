import sys
from datetime import datetime
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from DeepLearning import DeepLearning
from UIMain import Ui_MainWindow
from qt_material import apply_stylesheet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
import numpy as np


class MyClass(QMainWindow, Ui_MainWindow):
    Ver = "版本：1.0.0.4       更新日期：20230404"
    dl = DeepLearning()

    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent=parent)
        self.init()
        # 绑定深度学习类中textWritten信号，此信号来刷新Log
        self.dl.textWritten.connect(self.write_text)
        # 刷新底部状态栏显示相关信息
        self.statusBar().showMessage(self.Ver)

    # 初始化控件&&绑定控件事件
    def init(self):
        self.setupUi(self)
        self.write_text(' 软件初始化成功')
        self.setWindowTitle('基于特征识别的手写数字识别')
        self.setWindowIcon(QIcon('./logo.ico'))

        # 构建Matplolib，并添加到控件
        self.fig = plt.Figure()
        self.canvas = FC(self.fig)
        self.verticalLayout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self.ax.set(xlim=[0.5, 4.5], ylim=[-2, 8], title='Loss',
               ylabel='Y-Axis', xlabel='X-Axis')




        # 训练开始按钮绑定
        self.Button_Train.clicked.connect(self.run)
        self.Button_LoadModel.clicked.connect(self.DrawMat)


    # 这里是对深度学习训练运行的调用
    def run(self):
        self.dl.start()

    def DrawMat(self):
        self.ax.cla()
        x = np.linspace(0, 100, 100)
        y = np.random.random(100)
        self.ax.plot(x, y)
        self.canvas.draw()




    # Log记录
    def write_text(self, str):
        now = datetime.now()
        Mes = now.strftime("%H:%M:%S") + str
        self.textEdit.append(Mes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mc = MyClass()
    apply_stylesheet(app, theme='light_cyan_500.xml')
    Mc.show()
    sys.exit(app.exec_())
