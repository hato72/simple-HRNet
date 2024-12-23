# # gait_phase_detector.py
# import numpy as np
# from collections import deque

# class GaitPhaseDetector:
#     def __init__(self, window_size=5, height_threshold=3):
#         self.window_size = window_size
#         self.height_threshold = height_threshold
#         self.left_ankle_heights = deque(maxlen=window_size)
#         self.right_ankle_heights = deque(maxlen=window_size)
#         self.left_phase = None  # "stance" or "swing"
#         self.right_phase = None
        
#     def detect_phase_change(self, keypoints):
#         # COCOフォーマットでの足首のインデックス
#         LEFT_ANKLE_IDX = 15
#         RIGHT_ANKLE_IDX = 16
        
#         # 足首の位置を取得
#         left_ankle = keypoints[LEFT_ANKLE_IDX]
#         right_ankle = keypoints[RIGHT_ANKLE_IDX]
        
#         # y座標を記録 (画像座標系なので、値が大きいほど下)
#         self.left_ankle_heights.append(left_ankle[1])
#         self.right_ankle_heights.append(right_ankle[1])
        
#         phase_changes = []
        
#         if len(self.left_ankle_heights) == self.window_size:
#             # 移動平均による平滑化
#             left_heights = np.array(self.left_ankle_heights)
#             right_heights = np.array(self.right_ankle_heights)
            
#             # 傾きを計算
#             left_slope = np.mean(np.diff(left_heights))
#             right_slope = np.mean(np.diff(right_heights))
            
#             # 左足の判定
#             if self.left_phase == "swing" and left_slope > self.height_threshold:
#                 # 下降傾向で接地と判定
#                 phase_changes.append(("left", "touchdown"))
#                 self.left_phase = "stance"
#             elif self.left_phase == "stance" and left_slope < -self.height_threshold:
#                 # 上昇傾向で離地と判定
#                 phase_changes.append(("left", "takeoff"))
#                 self.left_phase = "swing"
#             elif self.left_phase is None:
#                 self.left_phase = "stance"
            
#             # 右足の判定
#             if self.right_phase == "swing" and right_slope > self.height_threshold:
#                 phase_changes.append(("right", "touchdown"))
#                 self.right_phase = "stance"
#             elif self.right_phase == "stance" and right_slope < -self.height_threshold:
#                 phase_changes.append(("right", "takeoff"))
#                 self.right_phase = "swing"
#             elif self.right_phase is None:
#                 self.right_phase = "stance"
                
#         return phase_changes

import numpy as np
from collections import deque

class GaitPhaseDetector:
    def __init__(self, touchdown_angle=100, takeoff_angle=175, angle_tolerance=10):
        self.touchdown_angle = touchdown_angle
        self.takeoff_angle = takeoff_angle
        self.angle_tolerance = angle_tolerance
        self.sequence_number = 1
        self.last_state = None
        self.min_frames_between_detections = 10
        self.frames_since_last_detection = 0
        
    def calculate_knee_angle(self, keypoints):
        # COCOフォーマットでのインデックス
        RIGHT_HIP = 12
        RIGHT_KNEE = 14
        RIGHT_ANKLE = 16
        
        hip = keypoints[RIGHT_HIP]
        knee = keypoints[RIGHT_KNEE]
        ankle = keypoints[RIGHT_ANKLE]
        
        # 3点から角度を計算
        vector1 = [hip[0] - knee[0], hip[1] - knee[1]]
        vector2 = [ankle[0] - knee[0], ankle[1] - knee[1]]
        
        # 内積とベクトルの大きさから角度を計算
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
        
        cos_angle = dot_product / (magnitude1 * magnitude2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        return angle
        
    def detect_phase_change(self, keypoints):
        self.frames_since_last_detection += 1
        
        # 膝関節角度を計算
        knee_angle = self.calculate_knee_angle(keypoints)
        
        # 最小フレーム間隔をチェック
        if self.frames_since_last_detection < self.min_frames_between_detections:
            return None, knee_angle
            
        # 接地判定（膝角度が約100度）
        if (abs(knee_angle - self.touchdown_angle) < self.angle_tolerance and 
            self.last_state != "touchdown"):
            self.last_state = "touchdown"
            self.frames_since_last_detection = 0
            return (self.sequence_number, "touchdown"), knee_angle
            
        # 離地判定（膝角度が約180度）
        elif (abs(knee_angle - self.takeoff_angle) < self.angle_tolerance and 
              self.last_state != "takeoff"):
            self.last_state = "takeoff"
            self.frames_since_last_detection = 0
            self.sequence_number += 1
            return (self.sequence_number, "takeoff"), knee_angle
            
        return None, knee_angle




# import numpy as np
# from collections import deque
# from enum import Enum

# class GaitPhase(Enum):
#     INITIAL_CONTACT = 1    # 初期接地
#     LOADING_RESPONSE = 2   # 荷重応答期
#     MID_STANCE = 3        # 立脚中期
#     TERMINAL_STANCE = 4   # 立脚終期
#     PRE_SWING = 5         # 前遊脚期
#     INITIAL_SWING = 6     # 遊脚初期
#     MID_SWING = 7         # 遊脚中期
#     TERMINAL_SWING = 8    # 遊脚終期

# class GaitPhaseDetector:
#     def __init__(self, 
#                  window_size=10, 
#                  velocity_threshold=3.0,
#                  acceleration_threshold=2.0,
#                  min_phase_frames=5):
#         """
#         Parameters:
#         -----------
#         window_size: int
#             移動平均を計算するためのウィンドウサイズ
#         velocity_threshold: float
#             速度変化を検出する閾値
#         acceleration_threshold: float
#             加速度変化を検出する閾値
#         min_phase_frames: int
#             各相の最小フレーム数
#         """
#         self.window_size = window_size
#         self.velocity_threshold = velocity_threshold
#         self.acceleration_threshold = acceleration_threshold
#         self.min_phase_frames = min_phase_frames
        
#         # 足首の位置履歴
#         self.left_ankle_history = deque(maxlen=window_size)
#         self.right_ankle_history = deque(maxlen=window_size)
        
#         # 現在の歩行相
#         self.left_phase = None
#         self.right_phase = None
        
#         # フレームカウンター
#         self.left_frame_counter = 0
#         self.right_frame_counter = 0
        
#         # 前回のイベント時刻
#         self.last_left_event = None
#         self.last_right_event = None

#     def calculate_kinematics(self, positions):
#         """位置データから速度と加速度を計算"""
#         if len(positions) < 3:
#             return None, None
            
#         positions = np.array(positions)
        
#         # 速度計算 (1次微分)
#         velocities = np.diff(positions, axis=0)
        
#         # 加速度計算 (2次微分)
#         accelerations = np.diff(velocities, axis=0)
        
#         return velocities[-1], accelerations[-1]

#     def detect_initial_contact(self, ankle_pos, ankle_vel, ankle_acc):
#         """初期接地（足が地面に触れ始める瞬間）の検出"""
#         # 下向きの動きが減速し始める点を検出
#         return (ankle_vel[1] > 0 and  # y方向の速度が正（下向き）
#                 ankle_acc[1] < -self.acceleration_threshold and  # 急な減速
#                 abs(ankle_vel[0]) < self.velocity_threshold)  # x方向の動きが小さい

#     def detect_toe_off(self, ankle_pos, ankle_vel, ankle_acc):
#         """離地（かかとが地面から離れる瞬間）の検出"""
#         # 上向きの加速が始まる点を検出
#         return (ankle_vel[1] < 0 and  # y方向の速度が負（上向き）
#                 ankle_acc[1] < -self.acceleration_threshold and  # 上向きの加速
#                 abs(ankle_vel[0]) < self.velocity_threshold)  # x方向の動きが小さい

#     def detect_phase_change(self, keypoints):
#         """
#         歩行相の変化を検出
        
#         Parameters:
#         -----------
#         keypoints: np.ndarray
#             COCOフォーマットのキーポイント配列
            
#         Returns:
#         --------
#         list of tuples
#             検出されたイベント (side, event_type, confidence)
#         """
#         LEFT_ANKLE_IDX = 15
#         RIGHT_ANKLE_IDX = 16
        
#         left_ankle = keypoints[LEFT_ANKLE_IDX]
#         right_ankle = keypoints[RIGHT_ANKLE_IDX]
        
#         # 信頼度スコアの確認
#         if left_ankle[2] < 0.3 or right_ankle[2] < 0.3:
#             return []
        
#         # 位置履歴の更新
#         self.left_ankle_history.append(left_ankle[:2])
#         self.right_ankle_history.append(right_ankle[:2])
        
#         events = []
        
#         # 十分なデータが蓄積された場合に解析
#         if len(self.left_ankle_history) == self.window_size:
#             # 左足の解析
#             left_vel, left_acc = self.calculate_kinematics(self.left_ankle_history)
#             if left_vel is not None:
#                 left_pos = np.array(self.left_ankle_history[-1])
                
#                 # 初期接地の検出
#                 if (self.detect_initial_contact(left_pos, left_vel, left_acc) and 
#                     self.left_frame_counter >= self.min_phase_frames):
#                     events.append(("left", "initial_contact", left_ankle[2]))
#                     self.left_frame_counter = 0
                
#                 # 離地の検出
#                 elif (self.detect_toe_off(left_pos, left_vel, left_acc) and 
#                       self.left_frame_counter >= self.min_phase_frames):
#                     events.append(("left", "toe_off", left_ankle[2]))
#                     self.left_frame_counter = 0
                
#                 self.left_frame_counter += 1
            
#             # 右足の解析（左足と同様）
#             right_vel, right_acc = self.calculate_kinematics(self.right_ankle_history)
#             if right_vel is not None:
#                 right_pos = np.array(self.right_ankle_history[-1])
                
#                 if (self.detect_initial_contact(right_pos, right_vel, right_acc) and 
#                     self.right_frame_counter >= self.min_phase_frames):
#                     events.append(("right", "initial_contact", right_ankle[2]))
#                     self.right_frame_counter = 0
                
#                 elif (self.detect_toe_off(right_pos, right_vel, right_acc) and 
#                       self.right_frame_counter >= self.min_phase_frames):
#                     events.append(("right", "toe_off", right_ankle[2]))
#                     self.right_frame_counter = 0
                
#                 self.right_frame_counter += 1
        
#         return events