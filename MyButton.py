from PyQt5.QtWidgets import QLabel,QRubberBand,QApplication,QMainWindow,QVBoxLayout,QPushButton,QWidget,QErrorMessage,QMessageBox,QDialog,QScrollBar,QSlider, QFileDialog ,QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsPixmapItem, QVBoxLayout, QWidget, QPushButton, QColorDialog
from PyQt5 import QtWidgets, uic, QtCore,QtGui

class MyButton(QtWidgets.QPushButton):
    def __init__(self, *args, **kwargs):
        super(MyButton, self).__init__(*args, **kwargs)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap

    def sizeHint(self):
        parent_size = QtWidgets.QPushButton.sizeHint(self)
        return QtCore.QSize(parent_size.width() + self.pixmap.width(), max(parent_size.height(), self.pixmap.height()))

    def paintEvent(self, event):
        QtWidgets.QPushButton.paintEvent(self, event)

        pos_x = 5  # hardcoded horizontal margin
        pos_y = (self.height() - self.pixmap.height())/2
        
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(pos_x, int(pos_y), self.pixmap)