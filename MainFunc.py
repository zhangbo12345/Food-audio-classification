import os
import sys

# qt5所需包
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QPixmap
from MainForm import *

#音频分类包
from first_get_feature import get_npy,get_logmfcc_pic
from second_easy_cnn import get_model
from third_predict import get_pre
class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)

        #显示背景
        self.qlabel_background.setPixmap(QPixmap('./background.jpg'))
        self.qlabel_background.setScaledContents (True)
        #显示训练曲线
        self.qlabel_picture.setPixmap(QPixmap('./filename.png'))
        self.qlabel_picture.setScaledContents(True)

    def extract_feature(self):
        print('extract_feature')
        get_npy()
        self.show_massage('特征已提取完毕')

    def show_feature(self):
        print('show_feature')
        file = self.open_file()
        get_logmfcc_pic(file)
        self.show_pic('./feature.png')

    def train_model(self):
        print('train_model')
        get_model()
        self.show_massage('模型以训练文完毕')

    def pre_single(self):
        print('pre_single')
        file = self.open_file()
        result = get_pre(file)
        self.show_massage('The result of this clip is:')
        self.show_massage(result)

    def show_massage(self,text):
        self.qTE_show_mess.append(str(text))

    def show_pic(self,pic):
        self.qlabel_picture.setPixmap(QPixmap(pic))
        self.show_massage('Picture have been loaded')

    def open_file(self):
        fileName, fileType = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", os.getcwd(),
                                            "All Files(*);;Text Files(*.wav)")
        print(fileName)
        print(fileType)
        return fileName


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())