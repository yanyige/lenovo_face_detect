# -*- coding: utf-8 -*-
"""
@File    : main.py
@Author  : yanyige
@Mail    : yige.yan@qq.com
@Time    : 2025/7/28 10:42
@Desc    : 主入口
"""

import sys
from threading import Thread
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from window import Ui_Form
from config import config
import myframe
import atexit
import time
from playsound import playsound


class MainWindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__()
        self.setupUi(self)
        self.init_ui()
        self.camera = None

    def init_ui(self):
        self.pushButton.clicked.connect(self.reset_counters)
        self.init_camera()

    def reset_counters(self):
        """重置所有计数器"""
        if hasattr(self, 'camera') and self.camera:
            self.camera.fatigue_detector.reset_counters()
            self._update_counter_ui()
            # 重置后立即更新状态显示
            self.camera._update_fatigue_status('normal')

    def _update_counter_ui(self):
        """更新计数器UI"""
        if not hasattr(self, 'camera') or not self.camera:
            return

        detector = self.camera.fatigue_detector

        # 更新眼睛计数
        self.label_6.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.total_eye_closed}</span></p ></body></html>")

        # 更新嘴巴计数
        self.label_16.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.total_mouth_open}</span></p ></body></html>")

    def update_counters(self):
        try:
            counters = [
                (self.label_15, config.TOTAL),
                (self.label_16, config.mTOTAL),
                (self.label_17, config.hTOTAL),
            ]
            for label, value in counters:
                label.setText(
                    f"<html><head/><body><p align='center'>"
                    f"<span style='font-size:10pt;font-weight:600;'>"
                    f"{value}</span></p></body></html>"
                )
        except Exception as e:
            print("更新计数器失败:", e)

    def init_camera(self, retry_count=3):
        for i in range(retry_count):
            try:
                self.camera = CameraController(self)
                return
            except Exception as e:
                if i == retry_count - 1:
                    QtWidgets.QMessageBox.critical(
                        self, "错误", f"初始化摄像头失败: {e}"
                    )
                else:
                    print(f"初始化摄像头失败: {e}")
                    QtCore.QThread.sleep(1)

    def closeEvent(self, event):
        if self.camera:
            self.camera.release()
        event.accept()

class AdaptiveThresholdCalculator:
    def __init__(self, init_eye_thresh=0.26):
         # 初始阈值
        self.eye_thresh = init_eye_thresh
        # 历史数据缓冲区
        self.eye_buffer = []
        self.buffer_size = 30  # 保留最近30帧数据

        # 自适应参数
        self.eye_adjust_factor = 0.05  # 阈值调整幅度
        self.min_eye_thresh = 0.15  # 最小阈值
        self.max_eye_thresh = 0.35 #允许的最高阈值
    
    def update_thresholds(self, eye_ar):
        # 1.维护缓冲区大小：移除最旧的数据
        self.eye_buffer.append(eye_ar)
        # 2. 维护缓冲区大小：移除最旧的数据
        if len(self.eye_buffer) > self.buffer_size:
            self.eye_buffer.pop(0)
        # 3. 当缓冲区满时进行阈值调整
        if len(self.eye_buffer) == self.buffer_size:
            # 计算最近30帧的平均EAR
            eye_mean = np.mean(self.eye_buffer)
            # 4a. 如果平均EAR明显低于当前阈值（用户眼睛较小）
            if eye_mean < self.eye_thresh - 0.1:
                self.eye_thresh = max(self.min_eye_thresh,
                                    self.eye_thresh - self.eye_adjust_factor)
            # 4b. 如果平均EAR明显高于当前阈值（用户眼睛较大）
            elif eye_mean > self.eye_thresh + 0.1:
                self.eye_thresh = min(self.max_eye_thresh,
                                    self.eye_thresh + self.eye_adjust_factor)
        # 5. 返回更新后的阈值
        return self.eye_thresh



class FatigueDetector:
    def __init__(self):
        # 眼睛参数
        # self.EYE_AR_THRESH = 0.26  # 眼睛长宽比阈值
        # 使用自适应阈值计算器
        self.threshold_calculator = AdaptiveThresholdCalculator()

        self.EYE_AR_CONSEC_FRAMES = 2  # 连续帧阈值

        # 嘴巴参数
        self.MAR_THRESH = 0.65  # 打哈欠长宽比阈值
        self.MOUTH_AR_CONSEC_FRAMES = 3  # 连续帧阈值

        # 头部姿态参数
        self.HAR_THRESH = 0.3  # 点头阈值
        self.NOD_AR_CONSEC_FRAMES = 3  # 连续帧阈值

        # 疲劳检测周期，改为时间基础而非帧数基础
        self.FATIGUE_CHECK_INTERVAL = 2.0  # 每2秒评估一次疲劳状态
        self.STATUS_MAINTAIN_DURATION = 30.0  # 状态维持30秒

        # 疲劳判断阈值
        self.FATIGUE_THRESHOLD = 0.04  # 疲劳阈值 (原值为0.02，过低导致误报)
        self.WARNING_THRESHOLD = 0.02  # 警告阈值 (原值为0.25)

        # 初始化时间戳
        self.last_check_time = time.time()  # 上次评估时间
        self.last_fatigue_time = 0  # 记录最后一次疲劳状态的时间
        self.last_warning_time = 0  # 记录最后一次警告状态的时间

        # 初始化计数器
        self.reset_counters()

    def reset_counters(self):
        """重置所有计数器"""
        # 眼睛计数
        self.eye_counter = 0
        self.total_eye_closed = 0
        self.eye_cycle_count = 0

        # 嘴巴计数
        self.mouth_counter = 0
        self.total_mouth_open = 0
        self.mouth_cycle_count = 0

        # 点头计数
        self.nod_counter = 0
        self.total_nod = 0
        self.nod_cycle_count = 0

        # 分心行为计数
        self.action_counter = 0

        # 30秒统计
        self.thirty_sec_eye = 0
        self.thirty_sec_mouth = 0
        self.thirty_sec_nod = 0

        # 疲劳检测周期
        self.fatigue_cycle = 0

        # 重置状态时间
        self.last_fatigue_time = 0
        self.last_warning_time = 0
        self.last_check_time = time.time()

    def detect_fatigue(self, eye_ar: float, mouth_ar: float) -> str:
        """
        检测疲劳状态
        返回:
        - "fatigue": 疲劳
        - "warning": 警告
        - "normal": 正常
        """
        # 更新自适应阈值
        self.EYE_AR_THRESH = self.threshold_calculator.update_thresholds(eye_ar)
        # 1. 眼睛闭合检测
        print(eye_ar)
        if eye_ar < self.EYE_AR_THRESH:
            self.eye_counter += 1  # 增加连续闭眼帧计数器
            self.eye_cycle_count += 1  # 增加当前疲劳周期内的闭眼帧计数
        else:
            if (
                self.eye_counter >= self.EYE_AR_CONSEC_FRAMES
            ):  # 检查连续闭眼帧数是否达到阈值
                self.total_eye_closed += 1  # 增加总闭眼次数
                self.thirty_sec_eye += 1  # 增加两分钟统计周期内的闭眼次数
            self.eye_counter = 0  # 重置连续闭眼帧计数器

        # 2. 打哈欠检测
        if mouth_ar > self.MAR_THRESH:  # 如果嘴巴张开
            self.mouth_counter += 1  # 增加连续张嘴帧计数器
            self.mouth_cycle_count += 1  # 增加当前疲劳周期内的张嘴帧计数
        else:
            if (
                self.mouth_counter >= self.MOUTH_AR_CONSEC_FRAMES
            ):  # 检查连续张嘴帧数是否达到阈值
                self.total_mouth_open += 1  # 增加总打哈欠次数
                self.thirty_sec_mouth += 1  # 增加两分钟统计周期内的打哈欠次数
            self.mouth_counter = 0

        # 3：重构疲劳判断逻辑（基于时间而非帧数）
        current_time = time.time()
        current_status = "normal"  # 默认状态

        # 每2秒评估一次疲劳状态
        if current_time - self.last_check_time >= self.FATIGUE_CHECK_INTERVAL:
            # 使用实际时间间隔计算疲劳分数
            time_interval = max(0.1, current_time - self.last_check_time)  # 避免除以0
            fatigue_score = self._calculate_fatigue_score(time_interval)
            print(f"疲劳分数: {fatigue_score:.4f} (时间间隔: {time_interval:.2f}s)")
            self._reset_cycle_counters()
            self.last_check_time = current_time

            # 记录疲劳事件
            if fatigue_score > self.FATIGUE_THRESHOLD:
                self.last_fatigue_time = current_time
                print("检测到疲劳状态!")
            elif fatigue_score > self.WARNING_THRESHOLD:  # 警告阈值
                self.last_warning_time = current_time
                print("检测到警告状态!")

        # 4：状态维持逻辑（优先疲劳>警告>正常），检查是否需要维持状态
        if current_time - self.last_fatigue_time < self.STATUS_MAINTAIN_DURATION:
            return "fatigue"
        elif current_time - self.last_warning_time < self.STATUS_MAINTAIN_DURATION:
            return "warning"
        else:
            return current_status

    def _calculate_fatigue_score(self,time_interval: float) -> float:
        """计算疲劳分数"""
        # 计算最近时间间隔内的闭眼/张嘴比例
        # 假设帧率为50fps，实际帧数 = 时间间隔 * 50
        estimated_frames = max(1, time_interval * 50)  # 避免除以0
        # 2. 计算闭眼和张嘴的比例（归一化到[0, 1]区间）
        eye_ratio = min(1.0, self.eye_cycle_count / estimated_frames)
        mouth_ratio = min(1.0, self.mouth_cycle_count / estimated_frames)

        # 调试信息
        print(f"闭眼帧数: {self.eye_cycle_count}, 张嘴帧数: {self.mouth_cycle_count}, 估计帧数: {estimated_frames:.1f}")
        print(f"闭眼比例: {eye_ratio:.4f}, 张嘴比例: {mouth_ratio:.4f}")

        # 修改点4: 调整权重，眼睛比例占更大权重
        return 0.8 * eye_ratio + 0.2 * mouth_ratio

    def _reset_cycle_counters(self):
        """重置周期计数器"""
        self.eye_cycle_count = 0
        self.mouth_cycle_count = 0


class CameraController:
    def __init__(self, parent_window):
        self.parent = parent_window
        # 警告消息播放标志位
        self.warning_playing = False
        # 检测疲劳状态检测器
        self.fatigue_detector = FatigueDetector()

        self.timer = QTimer()
        self._cleaned = False
        # 初始化状态
        self.last_status = "normal"
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps_start_time = QtCore.QTime.currentTime()

        # 三重释放保障
        self.parent.destroyed.connect(self.release)
        atexit.register(self.release)

        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("无法打开摄像头")

            # 设置摄像头参数以提高性能
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.timer.timeout.connect(self.update_frame)
            # 使用更短的定时器间隔，让Qt自己处理帧率
            self.timer.start(16)  # 约60FPS的定时器

        except Exception as e:
            self.release()
            raise e

    # 更新摄像头帧
    def update_frame(self):
        try:
            success, frame = self.camera.read()
            if not success:
                print("无法读取摄像头帧")
                return

            # 确保图像格式正确
            if frame is None:
                print("摄像头返回空帧")
                return

            # 确保图像是8位格式
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # 图像处理流水线
            frame = cv2.flip(frame, 1)  # 水平翻转
            # 返回处理后的帧
            frame, ret = myframe.frametest(frame)

            if ret and len(ret) >= 3:
                eyear, mouthar = ret[1], ret[2]
                # 检测疲劳状态
                previous_status = self.last_status
                current_status = self.fatigue_detector.detect_fatigue(eyear, mouthar)
            else:
                eyear, mouthar = 0.3, 0.4
                current_status = 'normal'
                # 重置检测器状态
                self.fatigue_detector.eye_cycle_count = 0
                self.fatigue_detector.mouth_cycle_count = 0
                print("未检测到人脸，重置状态")
            
            # 记录当前状态
            self.last_status = current_status

            # 状态变化时立即更新UI
            if current_status != previous_status:
                self._update_fatigue_status(current_status)

                # 疲劳时播放警告 (防止重复播放)
                if current_status == 'fatigue' and not self.warning_playing:
                    self.warning_playing = True
                    Thread(target=self._play_warning).start()
                elif current_status != 'fatigue':
                    self.warning_playing = False

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB

            # 缩放处理
            label_size = self.parent.label_11.size()
            h, w = rgb_image.shape[:2]
            target_w = label_size.width()
            target_h = target_w * h // w

            if target_h != label_size.height():
                target_h = label_size.height()
                target_w = int(w * target_h / h)

            resized = cv2.resize(
                rgb_image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

            # 显示图像
            bytes_per_line = 3 * target_w
            q_image = QImage(
                resized.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888
            )

            self.parent.label_11.setPixmap(QPixmap.fromImage(q_image))
            self._update_statistics()

            # 计算实际FPS
            self.frame_count += 1
            current_time = QtCore.QTime.currentTime()
            elapsed = self.fps_start_time.msecsTo(current_time)
            if elapsed >= 1000:  # 每秒更新一次FPS
                actual_fps = self.frame_count * 1000 / elapsed
                print(f"实际FPS: {actual_fps:.2f}")
                self.frame_count = 0
                self.fps_start_time = current_time

        except Exception as e:
            print(f"处理摄像头帧失败: {e}")
            # 不要在这里调用release()，避免无限循环

    def _update_fatigue_status(self, status):
        """更新疲劳状态显示"""
        status_text = {
            'normal': '清醒',
            'warning': '疲劳警告',
            'fatigue': '疲劳状态'
        }

        self.parent.label_16.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{status_text[status]}</span></p ></body></html>")

        # 设置颜色
        color_map = {
            'normal': '85, 255, 127',
            'warning': '255, 170, 0',
            'fatigue': '255, 0, 0'
        }
        self.parent.label_16.setStyleSheet(
            "border:3px solid rgb(66, 132, 198);\n"
            f"background-color:rgb({color_map[status]})"
        )

        # 疲劳时播放警告
        if status == 'fatigue':
            Thread(target=self._play_warning).start()

        def _play_warning(self):
            """播放警告声音"""
            if not self.warning_playing:
                self.warning_playing = True
                try:
                    print("播放疲劳警告音频...")
                    playsound('voice.mp3')
                except Exception as e:
                    print(f"播放音频时出现错误: {e}")
                finally:
                    self.warning_playing = False

    def _update_statistics(self):
        """更新统计信息"""
        detector = self.fatigue_detector

        # 更新眼睛闭合次数
        self.parent.label_6.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.total_eye_closed}</span></p ></body></html>")

        # 更新打哈欠次数
        self.parent.label_12.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.total_mouth_open}</span></p ></body></html>")

        # 更新点头次数
        self.parent.label_14.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.total_nod}</span></p ></body></html>")

        # 更新30秒统计
        self._update_thirty_sec_stats()

    def _update_thirty_sec_stats(self):
        """更新30妙统计"""
        detector = self.fatigue_detector

        # 眼睛统计
        self.parent.label_7.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.thirty_sec_eye}</span></p ></body></html>")
        self.parent.label_7.setStyleSheet(
            "border:3px solid rgb(66, 132, 198);\n"
            f"background-color:rgb({self._get_alert_color(detector.thirty_sec_eye)})"
        )

        # 嘴巴统计
        self.parent.label_13.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.thirty_sec_mouth}</span></p ></body></html>")
        self.parent.label_13.setStyleSheet(
            "border:3px solid rgb(66, 132, 198);\n"
            f"background-color:rgb({self._get_alert_color(detector.thirty_sec_mouth)})"
        )

        # 点头统计
        self.parent.label_15.setText(
            f"<html><head/><body><p align='center'>"
            f"<span style='font-size:10pt;font-weight:600;'>"
            f"{detector.thirty_sec_nod}</span></p ></body></html>")
        self.parent.label_15.setStyleSheet(
            "border:3px solid rgb(66, 132, 198);\n"
            f"background-color:rgb({self._get_alert_color(detector.thirty_sec_nod)})"
        )

    def _get_alert_color(self, count: int) -> str:
        """根据计数获取警告颜色"""
        if count < 5:
            return "85, 255, 127"  # 绿色
        elif 5 <= count < 10:
            return "255, 170, 0"  # 橙色
        else:
            return "255, 0, 0"  # 红色

    # 释放资源
    def release(self):
        if self._cleaned:
            return

        print("释放摄像头资源")
        try:
            if hasattr(self, "timer"):
                self.timer.timeout.disconnect()
                self.timer.stop()
                self.timer = None

            if hasattr(self, "camera") and self.camera.isOpened():
                self.camera.release()
                self.camera = None
        except Exception as e:
            print(f"释放摄像头资源失败: {e}")
        finally:
            self._cleaned = True





def main():
    try:
        app = QtWidgets.QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
        return 1


if __name__ == "__main__":
    main()
