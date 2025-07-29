# -*- coding: utf-8 -*-
"""
@File    : myfatigue.py
@Author  : yanyige
@Mail    : yige.yan@qq.com
@Time    : 2025/7/29 9:42
@Desc    : 人脸检测绘制
"""
import dlib
import cv2
import imutils
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

# 获取面部特征索引
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("weights/shape_predictor_68_face_landmarks.dat")

# 计算眼睛的纵横比
def calculate_eye_aspect_ratio(eye):
    # 垂直距离
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # 水平距离
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 计算嘴巴的纵横比
def calculate_mouth_aspect_ratio(mouth):
    A= np.linalg.norm(mouth[2] - mouth[10])
    B= np.linalg.norm(mouth[4] - mouth[8])
    C= np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

# 检测人脸特征点并绘制
def draw_facial_features(frame, shape):
    try:
        if shape is None or len(shape) != 68:
            return frame
            
        # 绘制眼睛
        left_eye = shape[lStart:lEnd]
        # 绘制右眼
        right_eye = shape[rStart:rEnd]
        for eye in [left_eye, right_eye]:
            eye_hull = cv2.convexHull(eye)
            cv2.drawContours(frame, [eye_hull], -1, (0, 255, 0), 1)
        # 绘制嘴巴
        mouth = shape[mStart:mEnd]
        mouth_hull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 0, 255), 1)

        # 如果需要绘制特定线条，可以在这里添加
        for (start, end) in [(38, 40), (43, 47), (51, 57), (48, 54)]:
            if start < len(shape) and end < len(shape):
                cv2.line(frame, shape[start], shape[end], (0, 0, 255), 1)
        return frame
    except Exception as e:
        print(f"绘制面部特征异常: {e}")
        return frame

# 检测人脸
def detect_fatigue(frame):
    try:
        # 确保图像格式正确
        if frame is None:
            print("输入帧为空")
            return frame, 0.0, 0.0
            
        # 确保图像是8位BGR格式
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        # 确保图像是3通道
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
        # 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 确保灰度图像是8位
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
            
        # 检测人脸
        rects = detector(gray, 0)

        eyear = 0.0
        mouthar = 0.0
        shape = None

        for rect in rects:
            try:
                # 读取面部标志点，包含68个面部特征点位置信息
                shape = predictor(gray, rect)
                # 将面部标志点转换为NumPy数组，一个NumPy数组包含68个面部特征点位置信息
                shape = face_utils.shape_to_np(shape)
                
                # 计算两只眼睛的比例
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                eyear = (calculate_eye_aspect_ratio(leftEye) + calculate_eye_aspect_ratio(rightEye)) / 2.0
                # 计算嘴巴的比例
                mouth = shape[mStart:mEnd]
                mouthar = calculate_mouth_aspect_ratio(mouth)
                break
            except Exception as e:
                print(f"处理单个人脸时出错: {e}")
                continue

        if shape is not None and len(shape) == 68:
            frame = draw_facial_features(frame, shape)
        else:
            # 如果没有检测到人脸或特征点不完整，直接返回原帧
            print("未检测到完整的人脸特征点")

        return frame, eyear, mouthar
    except Exception as e:
        print(f"人脸检测异常: {e}")
        import traceback
        traceback.print_exc()
        return frame, 0.0, 0.0

# 检测眨眼和打哈欠
def detect_blinking(shape):
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    mouth = shape[mStart:mEnd]

    leftEAR = calculate_eye_aspect_ratio(leftEye)
    rightEAR = calculate_eye_aspect_ratio(rightEye)
