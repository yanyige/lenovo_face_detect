import sys
import cv2
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap

class SimpleTest(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("简单摄像头测试")
        self.resize(800, 600)
        
        # 创建布局
        layout = QtWidgets.QVBoxLayout()
        
        # 创建标签用于显示图像
        self.image_label = QtWidgets.QLabel()
        self.image_label.setStyleSheet("border: 2px solid blue;")
        self.image_label.setMinimumSize(400, 300)
        layout.addWidget(self.image_label)
        
        # 创建按钮
        self.start_button = QtWidgets.QPushButton("开始摄像头")
        self.start_button.clicked.connect(self.start_camera)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
        
        # 初始化摄像头
        self.camera = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def start_camera(self):
        if self.camera is None:
            print("启动摄像头...")
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                print("摄像头启动成功")
                self.timer.start(100)  # 10fps
                self.start_button.setText("停止摄像头")
            else:
                print("摄像头启动失败")
                self.camera = None
        else:
            print("停止摄像头...")
            self.timer.stop()
            self.camera.release()
            self.camera = None
            self.image_label.clear()
            self.start_button.setText("开始摄像头")
    
    def update_frame(self):
        try:
            if not self.camera or not self.camera.isOpened():
                return
                
            ret, frame = self.camera.read()
            if not ret:
                print("无法读取摄像头帧")
                return
            
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 获取标签尺寸
            label_size = self.image_label.size()
            if label_size.width() <= 0 or label_size.height() <= 0:
                return
            
            # 缩放图像
            h, w = rgb_frame.shape[:2]
            target_w = label_size.width()
            target_h = target_w * h // w
            
            if target_h > label_size.height():
                target_h = label_size.height()
                target_w = int(w * target_h / h)
            
            resized = cv2.resize(rgb_frame, (target_w, target_h))
            
            # 创建QImage
            bytes_per_line = 3 * target_w
            q_image = QImage(resized.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)
            
            if q_image.isNull():
                print("QImage创建失败")
                return
            
            # 创建QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                print("QPixmap创建失败")
                return
            
            self.image_label.setPixmap(pixmap)
            print("图像显示成功")
            
        except Exception as e:
            print(f"更新帧时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        if self.camera:
            self.camera.release()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = SimpleTest()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 