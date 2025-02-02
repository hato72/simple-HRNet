# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# import os

# def load_images(img1_path, img2_path):
#     img1 = cv2.imread(img1_path)
#     img2 = cv2.imread(img2_path)
#     # 画像サイズを統一
#     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
#     return img1, img2

# def extract_person_silhouette(img):
#     # グレースケール変換
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # 適応的二値化でより正確なシルエット抽出
#     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                  cv2.THRESH_BINARY, 11, 2)
    
#     # ノイズ除去強化
#     kernel = np.ones((7,7), np.uint8)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
#     # Cannyエッジ検出を追加
#     edges = cv2.Canny(thresh, 50, 150)
    
#     # 輪郭検出
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
#     if contours:
#         # 面積でフィルタリング
#         significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
#         if significant_contours:
#             max_contour = max(significant_contours, key=cv2.contourArea)
#             mask = np.zeros_like(gray)
#             cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
#             return mask, max_contour
#     return None, None

# def calculate_similarity(img1, img2):
#     mask1, contour1 = extract_person_silhouette(img1)
#     mask2, contour2 = extract_person_silhouette(img2)
    
#     if mask1 is None or mask2 is None:
#         return 0.0
    
#     # 形状マッチング
#     shape_similarity = 1 - cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I3, 0)
    
#     # 位置の差異を計算
#     m1 = cv2.moments(mask1)
#     m2 = cv2.moments(mask2)
    
#     if m1['m00'] != 0 and m2['m00'] != 0:
#         cx1 = int(m1['m10']/m1['m00'])
#         cy1 = int(m1['m01']/m1['m00'])
#         cx2 = int(m2['m10']/m2['m00'])
#         cy2 = int(m2['m01']/m2['m00'])
        
#         # 重心間の距離を正規化
#         max_distance = np.sqrt(img1.shape[0]**2 + img1.shape[1]**2)
#         position_diff = 1 - np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2) / max_distance
#     else:
#         position_diff = 0
    
#     # 最終スコア（形状70%、位置30%）
#     final_score = 0.7 * shape_similarity + 0.3 * position_diff
    
#     return final_score

# # def calculate_similarity(img1, img2):
# #     # グレースケールに変換
# #     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# #     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
# #     # SSIM計算
# #     score, _ = ssim(gray1, gray2, full=True)
# #     return score

# def get_image_pairs(label_dir, gait_dir):
#     pairs = []
#     # label_difディレクトリ内の画像をリストアップ
#     label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    
#     for label_file in label_files:
#         # ファイル名から番号とタイプ(down/takeoff)を抽出
#         num = label_file.split('_')[0]
#         img_type = label_file.split('_')[1].split('.')[0]
        
#         # 対応するgait画像のパスを構築
#         gait_file = f"gait_sequence_{num.zfill(3)}_{img_type}.jpg"
#         gait_path = os.path.join(gait_dir, gait_file)
        
#         if os.path.exists(gait_path):
#             pairs.append({
#                 'label_path': os.path.join(label_dir, label_file),
#                 'gait_path': gait_path
#             })
    
#     return pairs

# if __name__ == "__main__":
#     base_dir = os.path.dirname(__file__)
#     label_dir = os.path.join(base_dir, "label_dif")
#     gait_dir = os.path.join(base_dir, "gait_images")
    
#     # すべての画像ペアを取得
#     image_pairs = get_image_pairs(label_dir, gait_dir)
    
#     # 各ペアの類似度を計算
#     for pair in image_pairs:
#         img1, img2 = load_images(pair['label_path'], pair['gait_path'])
#         similarity = calculate_similarity(img1, img2)
        
#         # ファイル名を取得（パスから）
#         label_name = os.path.basename(pair['label_path'])
#         gait_name = os.path.basename(pair['gait_path'])
        
#         print(f"比較: {label_name} vs {gait_name}")
#         print(f"類似度: {similarity:.4f}")
#         print("-" * 50)


import cv2
import numpy as np
from pynput import keyboard
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='Path to video file')
    return parser.parse_args()

class VideoAnalyzer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = 0
        self.manual_frames = []
        self.delay = 200
        self.current_frame = None  # 現在のフレームを保持
        
        #self.capture_dir = "capture_b"
        #self.capture_dir = "capture_b2"
        #self.capture_dir = "capture_b4"
        # 動画ファイル名から保存ディレクトリ名を生成
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.capture_dir = f"capture_{video_name}"
        if not os.path.exists(self.capture_dir):
            os.makedirs(self.capture_dir)

    def on_press(self, key):
        if key == keyboard.Key.space and self.current_frame is not None:
            self.manual_frames.append(self.frame_count)
            print(f"Marked frame: {self.frame_count}")
            
            # 現在保持しているフレームをコピーして保存
            try:
                frame_copy = self.current_frame.copy()
                filename = os.path.join(self.capture_dir, f"frame_{self.frame_count}.jpg")
                cv2.imwrite(filename, frame_copy)
                print(f"Saved frame to {filename}")
            except Exception as e:
                print(f"Error saving frame: {e}")
        
        elif key == keyboard.Key.up:
            self.delay = max(30, self.delay - 30)
            print(f"Speed up: {self.delay}ms")
        elif key == keyboard.Key.down:
            self.delay = min(500, self.delay + 30)
            print(f"Speed down: {self.delay}ms")
    
    def get_manual_frames(self):
        return self.manual_frames

    def analyze(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            self.current_frame = frame  # フレームを保持
            cv2.imshow('Frame', frame)
            self.frame_count += 1
            
            if cv2.waitKey(self.delay) & 0xFF == ord('q'):
                break
        print(self.manual_frames)

    def predict_frame(self, frame):
        # ここにモデルによるフレーム予測の実装
        # 仮の実装として現在のフレーム番号を返す
        return self.frame_count
    
if __name__ == "__main__":
    args = parse_args()
    analyzer = VideoAnalyzer(args.video)
    analyzer.analyze()

# python3 cal_similarity.py --video movie2/b5.mp4