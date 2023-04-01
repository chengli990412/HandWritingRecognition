import sys
from datetime import datetime

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow

from DeepLearning import DeepLearning
from UIMain import Ui_MainWindow

from qt_material import apply_stylesheet


class MyClass(QMainWindow, Ui_MainWindow):
    Ver="版本：1.0.0.3       更新日期：20230401"

    dl = DeepLearning()
    def __init__(self,parent=None):
        super(MyClass, self).__init__(parent=parent)
        self.init()
        self.dl.textWritten.connect(self.write_text)
        self.statusBar().showMessage(self.Ver)
        

    # 初始化
    def init(self):
        self.setupUi(self)
        self.write_text(' 软件初始化成功')
        self.setWindowTitle('基于特征识别的手写数字识别')
        self.setWindowIcon(QIcon('./logo.ico'))
        self.Button_Train.clicked.connect(self.run)

    def run(self):
        self.dl.start()

    # Log记录
    def write_text(self, str):
        now = datetime.now()
        Mes = now.strftime("%H:%M:%S") + str
        self.textEdit.append(Mes)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    Mc = MyClass()
    apply_stylesheet(app, theme='dark_purple.xml')
    Mc.show()
    sys.exit(app.exec_())
