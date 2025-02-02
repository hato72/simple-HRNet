python3  scripts/live-demo.py --filename running.mp4

python3  scripts/live-demo.py --filename movie/a1.mp4


## タスク
股関節と膝関節の角度計測(たぶんできてる)
接地した瞬間と離れた瞬間を取得(できた)

股関節角度を正しく図れるようにする
接地した瞬間の膝と股関節理想角度を表示
離地した瞬間の膝と股関節の理想角度を表示
そこから大きく外れていたらアドバイスを表示(定型文)

https://qiita.com/relu/items/81349a9bc0517c0e862a


def check_joint_angles(phase_change, hip_angle, knee_angle):
    """関節角度をチェックして問題点を返す"""
    feedback = []
    
    if phase_change:
        sequence_num, event = phase_change
        if event == "takeoff":
            # 離地時の角度チェック
            if not 150 <= hip_angle <= 170:  # 160±10
                feedback.append(f"離地時の股関節角度: {hip_angle:.1f}° (目標: 160±10°)")
            if not 145 <= knee_angle <= 175:  # 160±15
                feedback.append(f"離地時の膝関節角度: {knee_angle:.1f}° (目標: 160±15°)")
                
        elif event == "down":
            # 接地時の角度チェック
            if not 100 <= knee_angle <= 140:  # 120±20
                feedback.append(f"接地時の膝関節角度: {knee_angle:.1f}° (目標: 120±20°)")
                
    return feedback

# main関数内の該当部分を修正
if len(pts) > 0:
    # 人物が検出された場合の処理
    keypoints = pts[0]  # 最初の検出人物のみ使用
    phase_change, knee_angle = phase_detector.detect_phase_change(keypoints)
    
    # 角度を右上に表示
    frame_with_info = frame.copy()
    
    # 右上の座標を計算
    text_x = frame.shape[1] - 250  # 右端から250ピクセル
    text_y_start = 30  # 上端から30ピクセル
    
    # 角度情報を表示
    cv2.putText(frame_with_info, 
                f"Knee Angle: {knee_angle:.1f}°", 
                (text_x, text_y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    cv2.putText(frame_with_info, 
                f"Hip Angle: {hip_angle:.1f}°", 
                (text_x, text_y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)

    # フィードバックを表示
    feedback = check_joint_angles(phase_change, hip_angle, knee_angle)
    for i, text in enumerate(feedback):
        cv2.putText(frame_with_info,
                    text,
                    (10, frame.shape[0] - 30 - i*30),  # 下端から順に表示
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)  # 赤色で表示

    # フェーズ変化の検出時に画像を保存
    if phase_change:
        sequence_num, event = phase_change
        # イベントが検出されたら画像を保存
        filename = f"gait_sequence_{sequence_num:03d}_{event}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame_with_info)
        print(f"Saved {filename}")
        print(f"  Knee angle: {knee_angle:.1f}°")
        print(f"  Hip angle: {hip_angle:.1f}°")
