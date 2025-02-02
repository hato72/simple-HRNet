import os
import sys
import argparse
import ast
import cv2
import time
import torch
from vidgear.gears import CamGear
import numpy as np
from GaitPhaseDetector import *

sys.path.insert(1, os.getcwd())
from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation
from misc.utils import find_person_id_associations,calculate_angles
from PIL import Image, ImageDraw, ImageFont


def main(camera_id, filename, hrnet_m, hrnet_c, hrnet_j, hrnet_weights, hrnet_joints_set, image_resolution,
         single_person, yolo_version, use_tiny_yolo, disable_tracking, max_batch_size, disable_vidgear, save_video,
         video_format, video_framerate, device, enable_tensorrt):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

    # print(device)

    image_resolution = ast.literal_eval(image_resolution)
    has_display = 'DISPLAY' in os.environ.keys() or sys.platform == 'win32'
    video_writer = None

    if filename is not None:
        rotation_code = check_video_rotation(filename)
        video = cv2.VideoCapture(filename)
        assert video.isOpened()
    else:
        rotation_code = None
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    if yolo_version == 'v3':
        if use_tiny_yolo:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3-tiny.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3-tiny.weights"
        else:
            yolo_model_def = "./models_/detectors/yolo/config/yolov3.cfg"
            yolo_weights_path = "./models_/detectors/yolo/weights/yolov3.weights"
        yolo_class_path = "./models_/detectors/yolo/data/coco.names"
    elif yolo_version == 'v5':
        # YOLOv5 comes in different sizes: n(ano), s(mall), m(edium), l(arge), x(large)
        if use_tiny_yolo:
            yolo_model_def = "yolov5n"  # this  is the nano version
        else:
            yolo_model_def = "yolov5m"  # this  is the medium version
        if enable_tensorrt:
            yolo_trt_filename = yolo_model_def + ".engine"
            if os.path.exists(yolo_trt_filename):
                yolo_model_def = yolo_trt_filename
        yolo_class_path = ""
        yolo_weights_path = ""
    else:
        raise ValueError('Unsopported YOLO version.')

    # フレームレートの取得と設定
    original_fps = video.get(cv2.CAP_PROP_FPS)
    target_fps = original_fps * 2  # 例: フレームレートを2倍に
    video.set(cv2.CAP_PROP_FPS, target_fps)

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_version=yolo_version,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
        enable_tensorrt=enable_tensorrt
    )

    phase_detector = GaitPhaseDetector(
        touchdown_angle=90,  # 接地時の目標角度
        takeoff_angle=165,   # 離地時の目標角度
        angle_tolerance=10   # 許容誤差
    )
    #save_dir = "gait_images"  # 保存ディレクトリ
    #save_dir = "a_result"  # 保存ディレクトリ
    #save_dir = "b1_result"  # 保存ディレクトリ
    save_dir = "b2_result"  # 保存ディレクトリ
    #save_dir = "b4_result"  # 保存ディレクトリ
    os.makedirs(save_dir, exist_ok=True)

    detection_threshold = 0.1  # 例: しきい値を下げる
    model.detector_confidence_thresh = detection_threshold

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    output_path = 'output_pose.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0
    t_start = time.time()
    frame_count = 0

    while True:
        t = time.time()

        if filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                t_end = time.time()
                print("\n Total Time: ", t_end - t_start)
                break
            if rotation_code is not None:
                frame = cv2.rotate(frame, rotation_code)
        else:
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)

        frame_count += 1

        if not disable_tracking:
            boxes, pts = pts

        if not disable_tracking:
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(next_person_id, len(pts) + next_person_id, dtype=np.int32)
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes, pts=pts, prev_boxes=prev_boxes, prev_pts=prev_pts, prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id, pose_alpha=0.2, similarity_threshold=0.4, smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)
            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids

        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame = draw_points_and_skeleton(frame, pt, joints_dict()[hrnet_joints_set]['skeleton'], person_index=pid,
                                             points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                             points_palette_samples=10)
            
            hip_angle, knee_angle = calculate_angles(pt)
            # 角度をフレームに表示
            # cv2.putText(frame, f'Hip Angle: {hip_angle:.1f}deg', 
            #         (int(pt[11][0]), int(pt[11][1])-10),  # 股関節の近く
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, f'Knee Angle: {knee_angle:.1f}deg',
            #         (int(pt[12][0]), int(pt[12][1])-10),  # 膝の近く
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # for box in boxes:
        #     cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,255,255),2)

        # def check_joint_angles(phase_change, hip_angle, knee_angle):
        #     """関節角度をチェックして問題点を返す"""
        #     feedback = []
            
        #     if phase_change:
        #         sequence_num, event = phase_change
        #         if event == "takeoff":
        #             # 離地時の角度チェック
        #             if not 150 <= hip_angle <= 170:  # 160±10
        #                 feedback.append(f"Hip joint angle at takeoff: {hip_angle:.1f} (Target: 150-170)")
        #             if not 145 <= knee_angle <= 175:  # 160±15
        #                 feedback.append(f"Knee joint angle at takeoff: {knee_angle:.1f} (Target: 145-175)")
                        
        #         elif event == "down":
        #             # 接地時の角度チェック
        #             if not 100 <= knee_angle <= 140:  # 120±20
        #                 feedback.append(f"Knee joint angle at touchdown: {knee_angle:.1f} (Target: 100-140)")
                        
        #     return feedback
        from typing import List
        def put_japanese_text(img: np.ndarray, text: str, org: tuple, font_size: int, color: tuple) -> np.ndarray:
            """日本語テキストを画像に描画する関数"""
            # テキストを□に変換せずに表示
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', font_size)
            draw.text(org, text, font=font, fill=color)
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


        # 理想的な角度の定数定義
        TAKEOFF_HIP_IDEAL = 160  # 離地時の股関節の理想角度
        TAKEOFF_KNEE_IDEAL = 160  # 離地時の膝関節の理想角度
        TOUCHDOWN_KNEE_IDEAL = 120  # 接地時の膝関節の理想角度

        def check_joint_angles(phase_change, hip_angle, knee_angle):
            """関節角度をチェックして詳細なフィードバックを返す"""
            feedback = []
            
            if phase_change:
                sequence_num, event = phase_change
                if event == "takeoff":
                    # 離地時の股関節角度チェック
                    if hip_angle < 140:  # かなり小さい
                        feedback.append(f"股関節の伸展が足りません（現在: {hip_angle:.1f}°）。もっと大きく股関節を伸ばしてください。")
                    elif hip_angle < 150:  # やや小さい
                        feedback.append(f"股関節をもう少し伸ばしてください（現在: {hip_angle:.1f}°）。")
                    elif hip_angle > 180:  # かなり大きい
                        feedback.append(f"股関節の伸展が強すぎます（現在: {hip_angle:.1f}°）。伸ばしすぎないように注意してください。")
                    elif hip_angle > 170:  # やや大きい
                        feedback.append(f"股関節の伸展をやや抑えてください（現在: {hip_angle:.1f}°）。")
                    
                    # 離地時の膝関節角度チェック
                    if knee_angle < 135:  # かなり小さい
                        feedback.append(f"膝の伸展が不十分です（現在: {knee_angle:.1f}°）。力強く伸ばしてください。")
                    elif knee_angle < 145:  # やや小さい
                        feedback.append(f"膝をもう少し伸ばしてください（現在: {knee_angle:.1f}°）。")
                    elif knee_angle > 185:  # かなり大きい
                        feedback.append(f"膝の伸展が強すぎます（現在: {knee_angle:.1f}°）。")
                    # elif knee_angle > 175:  # やや大きい
                    #     feedback.append(f"膝の伸展をやや抑えてください（現在: {knee_angle:.1f}°）。")
                        
                elif event == "down":
                    # 接地時の膝関節角度チェック
                    if knee_angle < 90:  # かなり小さい
                        feedback.append(f"着地時の膝の屈曲が大きすぎます（現在: {knee_angle:.1f}°）。膝を伸ばして着地してください。")
                    elif knee_angle < 100:  # やや小さい
                        feedback.append(f"着地時の膝をもう少し伸ばしてください（現在: {knee_angle:.1f}°）。")
                    elif knee_angle > 150:  # かなり大きい
                        feedback.append(f"着地時の膝が伸びすぎています（現在: {knee_angle:.1f}°）。もっと膝を曲げてください。")
                    elif knee_angle > 140:  # やや大きい
                        feedback.append(f"着地時にもう少し膝を曲げてください（現在: {knee_angle:.1f}°）。")
                        
            return feedback
        
        # def draw_feedback(image, feedback):
        #     img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #     draw = ImageDraw.Draw(img_pil)

        #     # Linux用の一般的な日本語フォントパスを試行
        #     font_paths = [
        #         "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        #         "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        #         "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf"
        #     ]

        #     font = None
        #     for path in font_paths:
        #         try:
        #             font = ImageFont.truetype(path, 20)
        #             break
        #         except OSError:
        #             continue

        #     if font is None:
        #         # フォントが見つからない場合はデフォルトフォントを使用
        #         font = ImageFont.load_default()

        #     y_offset = 10
        #     for message in feedback:
        #         draw.text((10, y_offset), message, font=font, fill=(255, 255, 255))
        #         y_offset += 30

        #     return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        if len(pts) > 0:
            # 人物が検出された場合の処理
            keypoints = pts[0]  # 最初の検出人物のみ使用
            phase_change,knee_angle,hip_angles = phase_detector.detect_phase_change(keypoints,frame_count)

            # 角度を右上に表示
            frame_with_info = frame.copy()
            
            # 右上の座標を計算
            text_x = frame.shape[1] - 250  # 右端から250ピクセル
            text_y_start = 30  # 上端から30ピクセル
            
            # 常に膝関節角度を表示
            frame_with_info = frame.copy()
            cv2.putText(frame_with_info, 
                      f"Knee Angle: {knee_angle:.1f}", 
                      (10, 30),  # 左上に固定
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      1, 
                      (0, 255, 0), 
                      2)
            cv2.putText(frame_with_info, 
                      f"Hip Angle: {hip_angles:.1f}", 
                      (10, 70),  # 膝関節角度の下に表示
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      1, 
                      (0, 255, 0), 
                      2)
            
            # フィードバックを表示
            # feedback = check_joint_angles(phase_change, hip_angles, knee_angle)
            # for i, text in enumerate(feedback):
            #     cv2.putText(frame_with_info,
            #             text,
            #             (10, frame.shape[0] - 30 - i*30),  # 下端から順に表示
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.7, (0, 0, 255), 2)  # 赤色で表示
            feedback = check_joint_angles(phase_change, hip_angles, knee_angle)
            # フィードバックの各行を表示（下部に配置）
            height = frame_with_info.shape[0]
            margin_bottom = 50
            line_height = 40

            # フィードバックの各行を下から順に表示
            for i, text in enumerate(reversed(feedback)):
                y_pos = height - margin_bottom - i * line_height
                frame_with_info = put_japanese_text(
                    frame_with_info, 
                    text,
                    (50, y_pos),
                    30,
                    (0, 0, 255)  # BGR
                )
            
            
            if phase_change:
                sequence_num, event = phase_change
                # イベントが検出されたら画像を保存
                filename = f"gait_sequence_{sequence_num:03d}_{event}.jpg"
                filepath = os.path.join(save_dir, filename)
                
                cv2.imwrite(filepath, frame_with_info)
                print(f"Saved {filename} - Knee angle: {knee_angle:.1f}")
                print(f"Saved {filename} - Hip angle: {hip_angles:.1f}")

        if has_display:
            cv2.imshow('frame.png', frame)
            k = cv2.waitKey(1)
            if k == 27:  # Esc button
                if disable_vidgear:
                    video.release()
                else:
                    video.stop()
                break
        else:
            cv2.imwrite('frame.png', frame)

        if pts is not None:
            for person in pts:
                # キーポイントの描画
                for i, pt in enumerate(person):
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                
                # 骨格線の描画（右足）
                pairs = [(12, 14), (14, 16)]  # 右股関節→右膝→右足首
                for pair in pairs:
                    pt1 = person[pair[0]]
                    pt2 = person[pair[1]]
                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), 
                            (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
        out.write(frame)

    # if save_video:
    #     output_video.release()
    video.release()
    #accuracy = phase_detector.compare_frames([173,184,188,198,203,211,216]) #b1
    accuracy = phase_detector.compare_frames([132,140,145,155,159,169]) #b2
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", "-d", help="open the camera with the specified id", type=int, default=0)
    parser.add_argument("--filename", "-f", help="open the specified video (overrides the --camera_id option)",
                        type=str, default=None)
    parser.add_argument("--hrnet_m", "-m", help="network model - 'HRNet' or 'PoseResNet'", type=str, default='HRNet')
    parser.add_argument("--hrnet_c", "-c", help="hrnet parameters - number of channels (if model is HRNet), "
                                                "resnet size (if model is PoseResNet)", type=int, default=48)
    parser.add_argument("--hrnet_j", "-j", help="hrnet parameters - number of joints", type=int, default=17)
    parser.add_argument("--hrnet_weights", "-w", help="hrnet parameters - path to the pretrained weights",
                        type=str, default="./weights/pose_hrnet_w48_384x288.pth")
    parser.add_argument("--hrnet_joints_set",
                        help="use the specified set of joints ('coco' and 'mpii' are currently supported)",
                        type=str, default="coco")
    parser.add_argument("--image_resolution", "-r", help="image resolution", type=str, default='(384, 288)')
    parser.add_argument("--single_person",
                        help="disable the multiperson detection (YOLOv3 or an equivalen detector is required for"
                             "multiperson detection)",
                        action="store_true")
    parser.add_argument("--yolo_version",
                        help="Use the specified version of YOLO. Supported versions: `v3` (default), `v5`.",
                        type=str, default="v3")
    parser.add_argument("--use_tiny_yolo",
                        help="Use YOLOv3-tiny in place of YOLOv3 (faster person detection) if `yolo_version` is `v3`."
                             "Use YOLOv5n(ano) in place of YOLOv5m(edium) if `yolo_version` is `v5`."
                             "Ignored if --single_person",
                        action="store_true")
    parser.add_argument("--disable_tracking",
                        help="disable the skeleton tracking and temporal smoothing functionality",
                        action="store_true")
    parser.add_argument("--max_batch_size", help="maximum batch size used for inference", type=int, default=16)
    parser.add_argument("--disable_vidgear",
                        help="disable vidgear (which is used for slightly better realtime performance)",
                        action="store_true")  # see https://pypi.org/project/vidgear/
    parser.add_argument("--save_video", help="save output frames into a video.", action="store_true")
    parser.add_argument("--video_format", help="fourcc video format. Common formats: `MJPG`, `XVID`, `X264`."
                                               "See http://www.fourcc.org/codecs.php", type=str, default='MJPG')
    parser.add_argument("--video_framerate", help="video framerate", type=float, default=30)
    parser.add_argument("--device", help="device to be used (default: cuda, if available)."
                                         "Set to `cuda` to use all available GPUs (default); "
                                         "set to `cuda:IDS` to use one or more specific GPUs "
                                         "(e.g. `cuda:0` `cuda:1,2`); "
                                         "set to `cpu` to run on cpu.", type=str, default=None)
    parser.add_argument("--enable_tensorrt",
                        help="Enables tensorrt inference for HRnet. If enabled, a `.engine` file is expected as "
                             "weights (`--hrnet_weights`). This option should be used only after the HRNet engine "
                             "file has been generated using the script `scripts/export-tensorrt-model.py`.",
                        action='store_true')

    args = parser.parse_args()
    main(**args.__dict__)
