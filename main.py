import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from UIMain import Ui_MainWindow
from DeepLearning import DeepLearning


# 版本：1.0.0.7
# 更新日期：20230331


class MyClass(QWidget, Ui_MainWindow):
    dl = DeepLearning()

    def __init__(self):
        super(MyClass, self).__init__()
        self.init()
        self.dl.textWritten.connect(self.write_text)

    # 初始化
    def init(self):
        self.setupUi(self)
        self.textEdit.append('软件初始化成功')
        self.setWindowTitle('基于特征识别的手写数字识别')
        self.Button_Train.clicked.connect(self.run)
        self.setWindowIcon(QIcon('./logo.ico'))

    def run(self):
        self.dl.run()

    def write_text(self, str):
        self.textEdit.append(str)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mc = MyClass()
    Mc.show()
    sys.exit(app.exec_())
