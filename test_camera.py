import sys
import cv2
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from window import Ui_Form

class TestWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.camera = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 初始化摄像头
        self.init_camera()
        
    def init_camera(self):
        try:
            print("测试：尝试打开摄像头...")
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                print("测试：摄像头打开成功")
                self.timer.start(100)  # 10fps
            else:
                print("测试：摄像头打开失败")
        except Exception as e:
            print(f"测试：摄像头初始化错误: {e}")
    
    def update_frame(self):
        try:
            if not self.camera or not self.camera.isOpened():
                return
                
            ret, frame = self.camera.read()
            if not ret:
                print("测试：无法读取摄像头帧")
                return
            
            print(f"测试：成功读取帧，尺寸: {frame.shape}")
            
            # 转换为RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 获取label_11的尺寸
            label_size = self.label_11.size()
            print(f"测试：label_11尺寸: {label_size.width()}x{label_size.height()}")
            
            if label_size.width() <= 0 or label_size.height() <= 0:
                print("测试：label_11尺寸无效")
                return
            
            # 缩放图像
            h, w = rgb_frame.shape[:2]
            target_w = label_size.width()
            target_h = target_w * h // w
            
            if target_h > label_size.height():
                target_h = label_size.height()
                target_w = int(w * target_h / h)
            
            print(f"测试：缩放尺寸: {target_w}x{target_h}")
            resized = cv2.resize(rgb_frame, (target_w, target_h))
            
            # 创建QImage
            bytes_per_line = 3 * target_w
            q_image = QImage(resized.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)
            
            if q_image.isNull():
                print("测试：QImage创建失败")
                return
            
            # 创建QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            if pixmap.isNull():
                print("测试：QPixmap创建失败")
                return
            
            self.label_11.setPixmap(pixmap)
            print("测试：图像显示成功")
            
        except Exception as e:
            print(f"测试：更新帧时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def closeEvent(self, event):
        if self.camera:
            self.camera.release()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 