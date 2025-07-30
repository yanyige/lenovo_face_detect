"""
@File    : myframe.py
@Author  : yanyige
@Mail    : yige.yan@qq.com
@Time    : 2025/7/28 15:32
@Desc    : 帧绘制
"""
import time
import cv2
import mydetect
import numpy as np
import myfatigue

_fps_state = {
    'frame_count': 0,
    'start_time': time.time(),
    'last_time': time.time(),
    'fps': 0,
}

def frametest(frame):
    global _fps_state

    # 图像处理流水线
    ret = []
    label_list = []
    if frame is not None and hasattr(frame, 'shape'):
        ret.append(frame)
    else:
        ret.append(None)

    try:
        # 确保输入帧格式正确
        if frame is None:
            print("输入帧为空")
            return frame, [None, 0.0, 0.0]
            
        # 确保图像是8位格式
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
            
        _fps_state['frame_count'] += 1
        current_time = time.time()

        # TODO frame

        # 处理返回的元组 (frame, eyear, mouthar)
        result = myfatigue.detect_fatigue(frame)

        # 如果返回的是元组，则提取帧和眼睛、嘴巴纵横比
        if isinstance(result, tuple) and len(result) == 3:
            frame, eyear, mouthar = result
            frame, label_list = detect_action(frame, label_list)
            # 打印纵横比
            print(f"眼睛纵横比: {eyear:.3f}, 嘴巴纵横比: {mouthar:.3f}")
            ret.append(label_list)
            # 可以在这里使用 eyear 和 mouthar 进行疲劳检测
            ret.append(round(eyear, 3))
            ret.append(round(mouthar, 3))
        else:
            # 如果返回的不是元组，说明没有检测到人脸
            frame = result
            print("未检测到人脸")

        # 每秒计算FPS
        if current_time - _fps_state['last_time'] >= 1:
            _fps_state['fps'] = _fps_state['frame_count'] / (current_time - _fps_state['start_time'])
            _fps_state['frame_count'] = 0
            _fps_state['last_time'] = current_time
            # 减少调试信息输出频率
            if _fps_state['fps'] < 10:  # 只在FPS较低时输出
                print(f"FPS: {_fps_state['fps']:.2f}")

        # 确保帧可写
        if not getattr(frame, 'flags', None) or not frame.flags.writeable:
            frame = frame.copy()

        # 添加FPS文本
        cv2.putText(frame,
            f'FPS: {_fps_state["fps"]:.2f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2)
        
        return frame, ret

    except Exception as e:
        print(f"frametest异常: {e}")
        import traceback
        traceback.print_exc()
        return frame, ret

# 调用模型检测
def detect_action(frame, labellist):
    action = mydetect.predict(frame)

    for label, prob, xyxy in action:
        print("label", label)
        labellist.append(label)
        
        # 置信度和标签
        text = label + str(prob)

        # 绘制框
        left = int(xyxy[0])
        top = int(xyxy[1])
        right = int(xyxy[2])
        bottom = int(xyxy[3])
        # 绘制框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # 绘制标签
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame, labellist