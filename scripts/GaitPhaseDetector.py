import numpy as np
from collections import deque

class GaitPhaseDetector:
    def __init__(self, touchdown_angle, takeoff_angle, angle_tolerance=10):
        self.touchdown_angle = touchdown_angle
        self.takeoff_angle = takeoff_angle
        self.angle_tolerance = angle_tolerance
        self.sequence_number = 1
        self.last_state = None
        self.min_frames_between_detections = 10
        self.frames_since_last_detection = 0
        self.frame_num = 0
        self.min_frames_between_detections = 3  # 検出間隔を短縮
        self.debug = True  # デバッグ用フラグ
        self.detected_frames = []
        self.prev_knee_angle = None
        self.prev_hip_angle = None
        self.current_sequence = 0
        self.last_event_frame = 0
        self.phase_changes = []
        self.min_frames_between_events = 10
        self.ankle_hip_x_threshold = 80
        
    # def calculate_angles(self, keypoints):
    #     # COCOフォーマットでのインデックス
    #     RIGHT_HIP = 12
    #     RIGHT_KNEE = 14
    #     RIGHT_ANKLE = 16
        
    #     hip = keypoints[RIGHT_HIP]
    #     knee = keypoints[RIGHT_KNEE]
    #     ankle = keypoints[RIGHT_ANKLE]
        
    #     # 3点から角度を計算
    #     vector1 = [hip[0] - knee[0], hip[1] - knee[1]]
    #     vector2 = [ankle[0] - knee[0], ankle[1] - knee[1]]
        
    #     # 内積とベクトルの大きさから角度を計算
    #     dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    #     magnitude1 = np.sqrt(vector1[0]**2 + vector1[1]**2)
    #     magnitude2 = np.sqrt(vector2[0]**2 + vector2[1]**2)
        
    #     # cos_angle = dot_product / (magnitude1 * magnitude2)
    #     # angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
    #     # return angle

    #     knee_angle = np.degrees(np.arccos(np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)))

    #     # 股関節角度の計算
    #     vertical = [0, -1]  # 上向きの垂直ベクトル
    #     thigh_vector = [knee[0] - hip[0], knee[1] - hip[1]]
        
    #     dot_product = vertical[0] * thigh_vector[0] + vertical[1] * thigh_vector[1]
    #     magnitude_thigh = np.sqrt(thigh_vector[0]**2 + thigh_vector[1]**2)
        
    #     hip_angle = np.degrees(np.arccos(np.clip(dot_product / magnitude_thigh, -1.0, 1.0)))
        
    #     # 大腿部が垂直線の左側にある場合の補正
    #     if thigh_vector[0] < 0:
    #         hip_angle = 360 - hip_angle

    #     return hip_angle, knee_angle

    def calculate_angles(self, keypoints): #角度計算方法を修正
        NECK = 1
        RIGHT_HIP = 12
        RIGHT_KNEE = 14
        RIGHT_ANKLE = 16
        
        neck = keypoints[NECK]
        hip = keypoints[RIGHT_HIP]
        knee = keypoints[RIGHT_KNEE]
        ankle = keypoints[RIGHT_ANKLE]
        
        # 膝関節角度の計算（既存のコード）
        knee_vector1 = [hip[0] - knee[0], hip[1] - knee[1]]
        knee_vector2 = [ankle[0] - knee[0], ankle[1] - knee[1]]
        
        dot_product_knee = knee_vector1[0] * knee_vector2[0] + knee_vector1[1] * knee_vector2[1]
        magnitude_knee1 = np.sqrt(knee_vector1[0]**2 + knee_vector1[1]**2)
        magnitude_knee2 = np.sqrt(knee_vector2[0]**2 + knee_vector2[1]**2)
        
        knee_angle = np.degrees(np.arccos(np.clip(dot_product_knee / (magnitude_knee1 * magnitude_knee2), -1.0, 1.0)))
        
        # 股関節角度の計算（上半身と大腿部のベクトル）
        trunk_vector = [neck[0] - hip[0], neck[1] - hip[1]]
        thigh_vector = [knee[0] - hip[0], knee[1] - hip[1]]
        
        dot_product_hip = trunk_vector[0] * thigh_vector[0] + trunk_vector[1] * thigh_vector[1]
        magnitude_trunk = np.sqrt(trunk_vector[0]**2 + trunk_vector[1]**2)
        magnitude_thigh = np.sqrt(thigh_vector[0]**2 + thigh_vector[1]**2)
        
        hip_angle = np.degrees(np.arccos(np.clip(dot_product_hip / (magnitude_trunk * magnitude_thigh), -1.0, 1.0)))
        
        # # 大腿部が体幹の左側にある場合の補正
        # cross_product = trunk_vector[0] * thigh_vector[1] - trunk_vector[1] * thigh_vector[0]
        # if cross_product < 0:
        #     hip_angle = 360 - hip_angle
        
        return hip_angle, knee_angle
    

    # def calculate_leg_angle(self, keypoints):
    #     """膝からくるぶしへの線分の角度を計算（水平を0度として右回り）"""
    #     knee = keypoints[14]  # 右膝のインデックス
    #     ankle = keypoints[16]  # 右くるぶしのインデックス
        
    #     # ベクトルの計算
    #     dx = ankle[0] - knee[0]
    #     dy = ankle[1] - knee[1]
        
    #     # 角度の計算（ラジアンから度に変換）
    #     angle = np.degrees(np.arctan2(dy, dx))
        
    #     # 角度を0-360の範囲に正規化
    #     if angle < 0:
    #         angle += 360
            
    #     return angle

    # def is_knee_horizontal(self, keypoints, tolerance=15):
    #     """膝の位置が水平に近いかを判定"""
    #     hip = keypoints[12]  # 右股関節のインデックス
    #     knee = keypoints[14]  # 右膝のインデックス
        
    #     # 膝の高さが股関節より下にあるか確認
    #     if knee[1] <= hip[1]:
    #         return False
        
    #     # 膝の水平度を確認
    #     ankle = keypoints[16]  # 右くるぶしのインデックス
    #     knee_angle = abs(np.degrees(np.arctan2(ankle[1] - knee[1], ankle[0] - knee[0])))
        
    #     return abs(knee_angle - 90) <= tolerance
    
    def calculate_leg_to_horizontal_angle(self, keypoints):
        """膝からくるぶしへの線分と水平線のなす角度を計算（左回り）"""
        knee = keypoints[14]  # 右膝のインデックス
        ankle = keypoints[16]  # 右くるぶしのインデックス
        
        # ベクトルの計算
        dx = ankle[0] - knee[0]
        dy = ankle[1] - knee[1]
        
        # 角度の計算（ラジアンから度に変換）- 左回りで計算
        angle = np.degrees(np.arctan2(-dy, dx))  # -dyで左回りに
        
        # # 角度を0-360の範囲に正規化
        # if angle < 0:
        #     angle += 360
        #print("angle",angle)
        return angle
        

    def detect_phase_change(self, keypoints, frame_count):
        self.frames_since_last_detection += 1
        self.frame_num += 1
        
        if keypoints is None or len(keypoints) < 17:
            return None, 0,0
            
        hip_angle,knee_angle = self.calculate_angles(keypoints)
        #leg_angle = self.calculate_leg_angle(keypoints)
       
        # デバッグ用出力
        # if self.debug:
        #     print(f"Current knee angle: {knee_angle:.1f}")
        #     print(f"Current hip angle: {hip_angle:.1f}")
        #     print(f"Last state: {self.last_state}")
        #     print(f"Frames: {self.frame_num}")

        print(f"xxxxx-------: {self.frame_num}")
        leg_angle = self.calculate_leg_to_horizontal_angle(keypoints)
        #print(f"leg_angle1: {leg_angle:.1f}")
        if self.frames_since_last_detection < self.min_frames_between_detections:
            if self.debug:
                print(f"Frames-------: {self.frame_num}")
                print(f"Current knee angle: {knee_angle:.1f}")
                print(f"Current hip angle: {hip_angle:.1f}")
                print(f"Last state: {self.last_state}")
                print(f"leg_angle: {leg_angle:.1f}")
            return None, knee_angle,hip_angle
        
        
        # 膝が水平で、脚の角度が80-120度の範囲内の場合に着地とみなす
        # is_touchdown = (self.is_knee_horizontal(keypoints) and 
        #                80 <= leg_angle <= 120 and 
        #                frame_count - self.last_event_frame > self.min_frames_between_events)
        # is_touchdown = (70 <= leg_angle <= 100 and 
        #            frame_count - self.last_event_frame > self.min_frames_between_events)


        # is_touchdown = (leg_angle <= 60 and hip_angle >= 60 and
        #            frame_count - self.last_event_frame > self.min_frames_between_events)
        
        # if is_touchdown:
        #     self.current_sequence += 1
        #     self.sequence_number += 1
        #     self.last_event_frame = frame_count
        #     self.phase_changes.append((frame_count, "down"))
        #     return (self.sequence_number, "down"), knee_angle, hip_angle

        # 離地判定（既存のロジック）
        # is_takeoff = (abs(knee_angle - self.takeoff_angle) <= self.angle_tolerance and
        #              frame_count - self.last_event_frame > self.min_frames_between_events)

        # if is_takeoff:
        #     self.last_event_frame = frame_count
        #     self.phase_changes.append((frame_count, "takeoff"))
        #     return (self.sequence_number, "takeoff"), knee_angle, hip_angle

        # self.prev_knee_angle = knee_angle
        # self.prev_hip_angle = hip_angle
        # return None, knee_angle, hip_angle
        
        #leg_angle2 = self.calculate_leg_to_horizontal_angle(keypoints)
        if self.debug:
            print(f"Frames-------: {self.frame_num}")
            print(f"Current knee angle: {knee_angle:.1f}")
            print(f"Current hip angle: {hip_angle:.1f}")
            print(f"Last state: {self.last_state}")
            print(f"leg_angle: {leg_angle:.1f}")
        # 着地判定の条件を微調整
        knee_range = 20  # 許容範囲を広げる
        if ((self.takeoff_angle - knee_range <= knee_angle <= self.takeoff_angle + knee_range) and
            self.last_state != "takeoff"):
            self.last_state = "takeoff"
            self.frames_since_last_detection = 0
            self.detected_frames.append(frame_count)  
            # print("Takeoff detected at takeoff frame", frame_count)
            # モデル出力：13,64,184
            # 20,65,108,158
            return (self.sequence_number, "takeoff"), knee_angle,hip_angle

        # elif (abs(knee_angle - self.touchdown_angle) < self.angle_tolerance and 
        #       self.last_state != "down"):
        elif (leg_angle <= 60 and 80 <= hip_angle <= 125 and 120 <= knee_angle <= 160 and
            self.last_state == "takeoff"):

            self.last_state = "down"
            self.frames_since_last_detection = 0
            self.sequence_number += 1
            self.detected_frames.append(frame_count)  
            # print("Takeoff detected at touchdown frame", frame_count)
            # モデル出力：10,52,142
            #12,51,96,142
            return (self.sequence_number, "down"), knee_angle,hip_angle
            
        return None, knee_angle,hip_angle
    
    def compare_frames(self, manual_frames, threshold=3):
        correct_detections = 0
        matched_frames = []  # デバッグ用：マッチしたフレームのペアを保存
        detect_frame = self.detected_frames.copy()
    
        for manual_frame in manual_frames:
            found_match = False
            for detected_frame in self.detected_frames:
                if abs(manual_frame - detected_frame) <= threshold:
                    correct_detections += 1
                    matched_frames.append((manual_frame, detected_frame))
                    self.detected_frames.remove(detected_frame)
                    found_match = True
                    break
            if not found_match:
                print(f"No match found for manual frame: {manual_frame}")
        
        accuracy = correct_detections / len(manual_frames) if manual_frames else 0
        print(f"Manual frames: {manual_frames}")
        print(f"Detected frames: {detect_frame}")
        print(f"Matched pairs: {matched_frames}")
        # print(f"Accuracy: {accuracy:.2%} ({correct_detections}/{len(manual_frames)})")
        return accuracy


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