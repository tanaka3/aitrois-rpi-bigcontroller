import argparse
import time
import os
import sys
from functools import lru_cache
import threading
import subprocess

import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

from famicom_controller import FamicomControllerState, DebouncedFamicomControllerState, FamicomControllerSender


class FamicomAIController:
    CAPTURE_FOLDER = "captures"
    def __init__(self, args):
        self.args = args
        self.CAMERA_PREVIEW = args.preview
        
        # スレッド間で共有するデータのロック
        self.state_lock = threading.Lock()
        
        # 初期化
        self.fc_sender = FamicomControllerSender('/dev/ttyAMA0')
        self.init_controller_states()
        
        # カメラの初期化
        self.init_cameras()
        
        # 表示用の変数
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_detections = []
        self.last_detections_cam2 = []
        
        # 最後に送信した状態
        self.last_state = None
        
        # 前回の処理時間を記録（処理頻度制御用）
        self.last_process_time = 0
        self.last_send_time = 0
        self.last_capture_time = 0
        # 60FPSに合わせた送信間隔（約16.67ms）
        self.send_interval = 1.0 / 60.0

        # フォルダが存在しない場合は作成
        os.makedirs(FamicomAIController.CAPTURE_FOLDER, exist_ok=True)
        cmd = f"find {FamicomAIController.CAPTURE_FOLDER} -type f | wc -l"
        result = subprocess.check_output(cmd, shell=True)
        self.file_count = int(result.strip())

    def init_controller_states(self):
        """コントローラー状態管理オブジェクトの初期化"""
        button_thresholds = {'A': (2, 2), 'B': (2, 2), 'RIGHT': (2, 2), 'LEFT':(2, 2)}
        
        self.debounced_state_cam1 = DebouncedFamicomControllerState(
            default_threshold_true=3, 
            default_threshold_false=3, 
            button_thresholds=button_thresholds
        )
        
        self.debounced_state_cam2 = DebouncedFamicomControllerState(
            default_threshold_true=3, 
            default_threshold_false=3, 
            button_thresholds=button_thresholds
        )
        
        self.debounced_state = DebouncedFamicomControllerState(
            default_threshold_true=3, 
            default_threshold_false=3, 
            button_thresholds=button_thresholds
        )

    def init_cameras(self):
        """カメラの初期化"""
        # メインカメラの初期化
        camera_id = self.get_camera_id(0)
        if camera_id is None:
            print("Error: AI Camera not found.")
            sys.exit(1)
        
        self.imx500 = IMX500(self.args.model, camera_id)
        self.picam2 = Picamera2(0)
        
        # モデルのインストール進捗表示
        self.imx500.show_network_fw_progress_bar()
        
        # カメラ2の初期化（必要な場合）
        self.imx500_cam2 = None
        self.picam2_cam2 = None
        
        if self.args.dual_camera:
            camera_id = self.get_camera_id(1)
            if camera_id is None:
                print("Warning: Second AI Camera not found. Continuing with single camera.")
                self.args.dual_camera = False
            else:
                self.imx500_cam2 = IMX500(self.args.model, camera_id)
                self.picam2_cam2 = Picamera2(1)
                self.imx500_cam2.show_network_fw_progress_bar()


        # カメラの設定と開始
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.args.fps}, 
            buffer_count=4
        )

        # カメラ開始
        self.picam2.start(config, show_preview=self.CAMERA_PREVIEW)
        
        if self.args.dual_camera and self.picam2_cam2:
            config2 = self.picam2_cam2.create_preview_configuration(
                main={"size": (2028, 1520)},  # 撮影画像サイズ
                controls={"FrameRate": self.args.fps}, 
                buffer_count=4
            )
            self.picam2_cam2.start(config2, show_preview=False)

        # コールバック設定
        self.picam2.pre_callback = self.pre_callback


    def capture_img(self):
        """カメラからの画像をキャプチャ"""
        try:
            if self.picam2_cam2:
                self.picam2_cam2.capture_file(f"{FamicomAIController.CAPTURE_FOLDER}/{self.file_count:06d}.jpg")
                self.file_count += 1               
        except Exception as e:
            print(f"Error capturing image: {e}")

    def pre_callback(self, request):
        """カメラフレームのコールバック関数（別スレッドで実行される）"""
        # プレビュー表示が必要な場合のみ描画処理を行う
        if self.CAMERA_PREVIEW:
            self.draw_detections(request)

    def send_controller_state(self):
        """コントローラー状態の送信処理"""
        # ロックを取得して安全に状態を取得
        with self.state_lock:
            state = self.debounced_state.get_debounced_state()
            # 状態が変わった場合のみ送信する
            if self.last_state is None or not state.is_equal_to(self.last_state):
                try:
                    self.fc_sender.send_command(state)
                    self.last_state = state
                except Exception as e:
                    print(f"Error sending command: {e}")

    def process_frame(self):
        """メインスレッドでのフレーム処理"""
        current_time = time.time()

        self.last_process_time = current_time
        
        # カメラ1の検出処理
        detections = self.parse_detections(self.imx500, self.picam2)
        if detections is not None:
            self.frame_count += 1
            raw_state_cam1 = self.parse_state(detections)
            with self.state_lock:
                self.debounced_state_cam1.update(raw_state_cam1)
                self.last_detections = detections
        
        # カメラ2の検出処理（存在する場合）
        if self.args.dual_camera and self.imx500_cam2 and self.picam2_cam2:
            detections_cam2 = self.parse_detections(self.imx500_cam2, self.picam2_cam2)
            if detections_cam2 is not None:
                self.frame_count += 1
                raw_state_cam2 = self.parse_state(detections_cam2)
                with self.state_lock:
                    self.debounced_state_cam2.update(raw_state_cam2)
                    self.last_detections_cam2 = detections_cam2
        
        # 最終的な状態の更新
        with self.state_lock:
            self.debounced_state.update_consecutive_sum(
                self.debounced_state_cam1, 
                self.debounced_state_cam2
            )
        
        # FPS計算（1秒ごとに更新）
        elapsed_time = current_time - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.start_time = current_time
            self.frame_count = 0
        
        # 60FPSに合わせてコントローラー状態を送信
        if current_time - self.last_send_time >= self.send_interval:
            self.send_controller_state()
            self.last_send_time = current_time
   
            
        # 画像キャプチャのスレッドを開始 (検出時のみとする)
        if self.args.capture and current_time - self.last_capture_time > 5.0:
            self.last_capture_time = current_time
            threading.Thread(target=self.capture_img).start()

    def parse_state(self, detections):
        """検出情報をコントローラ状態に変換"""
        # カテゴリIDと対応するボタンのマッピング
        CATEGORY_MAPPING = {
            0: ["A"],
            1: ["B"],
            2: ["DOWN"],
            3: ["LEFT"],
            4: ["RIGHT"],
            5: ["SELECT"],
            6: ["START"],
            7: ["UP"],
        }
        
        raw_state = FamicomControllerState()
        
        for detection in detections:
            if detection.category in CATEGORY_MAPPING:
                for attr in CATEGORY_MAPPING[detection.category]:
                    if hasattr(raw_state, attr):
                        setattr(raw_state, attr, True)
        
        return raw_state

    def parse_detections(self, imx500, picam2):
        """カメラからの検出情報を解析"""
        try:
            metadata = picam2.capture_metadata()
            
            # 閾値
            threshold = self.args.threshold
            
            # 検出結果の取得
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            
            # 推論結果がない場合
            if np_outputs is None:
                return None
            
            # 検出情報の整理
            boxes, scores, classes = np_outputs[0][0], np_outputs[2][0], np_outputs[1][0]
            
            # 閾値以上の検出情報の設定
            detections = []
            for box, score, category in zip(boxes, scores, classes):
                if score >= threshold:
                    detections.append(
                        Detection(
                            picam2.camera_idx, 
                            category, 
                            score, 
                            imx500.convert_inference_coords(box, metadata, picam2)
                        )
                    )
            
            return detections
        except Exception as e:
            print(f"Error in detection: {e}")
            return None

    def draw_detections(self, request, stream="main"):
        """検出情報とコントローラ状態の描画"""
        try:
            with MappedArray(request, stream) as m:
                # FPS表示
                fps_label = f"FPS:{self.fps:.2f}"
                cv2.putText(
                    m.array,
                    fps_label,
                    (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                
                # コントローラの描画
                self.draw_controller(m.array, 450, 20)
                
                # カメラ1の検出情報描画
                self.draw_detection(m, self.last_detections)
                
                # カメラ2の検出情報描画（存在する場合）
                if self.args.dual_camera:
                    self.draw_detection(m, self.last_detections_cam2)
        except Exception as e:
            print(f"Error drawing detections: {e}")

    def draw_detection(self, img, detections):
        """検出ボックスと情報の描画"""
        try:
            # ラベル取得
            labels = self.get_labels()
            
            for detection in detections:
                # ボックスサイズ取得
                x, y, w, h = detection.box
                
                # カテゴリ別の色設定
                color_map = {
                    0: (255, 0, 0),       # 赤 (Aボタン)
                    1: (0, 255, 0),       # 緑 (Bボタン)
                    2: (0, 0, 255),       # 青 (下方向)
                    3: (255, 165, 0),     # オレンジ (左方向)
                    4: (255, 223, 0),     # 黄 (右方向)
                    5: (0, 255, 255),     # シアン (セレクトボタン)
                    6: (255, 0, 255),     # マゼンタ (スタートボタン)
                    7: (128, 128, 128),   # グレー (上方向)
                }
                
                color = color_map.get(int(detection.category), (255, 255, 255))
                
                # 検出情報のラベル表示
                label = f"cam{int(detection.id)}: {labels[int(detection.category)]} ({detection.score:.2f})"
                cv2.putText(
                    img.array,
                    label,
                    (x + 5, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                
                # 検出ボックスの描画
                cv2.rectangle(img.array, (x, y), (x + w, y + h), color, 2)
        except Exception as e:
            print(f"Error in draw_detection: {e}")

    def draw_controller(self, img, x, y):
        """画面上にコントローラーを描画"""
        try:
            # 背景の長方形
            cv2.rectangle(img, (x + 0, y + 0), (x + 151, y + 67), (160, 50, 50), -1)
            
            # 中央の形状
            pts = np.array([
                [x + 7, y + 7], [x + 64, y + 7], [x + 64, y + 27], 
                [x + 144, y + 27], [x + 144, y + 60], [x + 7, y + 60]
            ], np.int32)
            cv2.fillPoly(img, [pts], (230, 220, 190))
            
            cv2.line(img, (x + 7, y + 48), (x + 144, y + 48), (0, 0, 0), 1)
            cv2.line(img, (x + 7, y + 52), (x + 144, y + 52), (0, 0, 0), 1)
            
            # パッド
            cv2.rectangle(img, (x + 26, y + 27), (x + 36, y + 54), (0, 0, 0), -1)
            cv2.rectangle(img, (x + 17, y + 36), (x + 44, y + 46), (0, 0, 0), -1)
            
            # 中央ボタン
            cv2.rectangle(img, (x + 52, y + 40), (x + 92, y + 54), (160, 50, 50), -1)
            cv2.rectangle(img, (x + 56, y + 45), (x + 67, y + 50), (0, 0, 0), -1)
            cv2.rectangle(img, (x + 77, y + 45), (x + 88, y + 50), (0, 0, 0), -1)
            
            # A / B ボタン
            cv2.circle(img, (x + 109, y + 47), 7, (0, 0, 0), -1)
            cv2.circle(img, (x + 127, y + 47), 7, (0, 0, 0), -1)
            
            # 現在の状態を安全に取得
            with self.state_lock:
                last_state = self.debounced_state.get_debounced_state()
                
            # ボタン押下状態の描画
            # 上
            if last_state.UP:
                cv2.rectangle(img, (x + 28, y + 29), (x + 34, y + 37), (255, 0, 0), -1)
            # 下
            if last_state.DOWN:
                cv2.rectangle(img, (x + 28, y + 44), (x + 34, y + 52), (255, 0, 0), -1)
            # 左
            if last_state.LEFT:
                cv2.rectangle(img, (x + 19, y + 38), (x + 27, y + 44), (255, 0, 0), -1)
            # 右
            if last_state.RIGHT:
                cv2.rectangle(img, (x + 35, y + 38), (x + 42, y + 44), (255, 0, 0), -1)
            # セレクト
            if last_state.SELECT:
                cv2.rectangle(img, (x + 58, y + 47), (x + 65, y + 48), (255, 0, 0), -1)
            # スタート
            if last_state.START:
                cv2.rectangle(img, (x + 79, y + 47), (x + 86, y + 48), (255, 0, 0), -1)
            # B
            if last_state.B:
                cv2.circle(img, (x + 109, y + 47), 5, (255, 0, 0), -1)
            # A
            if last_state.A:
                cv2.circle(img, (x + 127, y + 47), 5, (255, 0, 0), -1)
                
            # 送信FPS表示
            send_fps = 1.0 / self.send_interval if self.send_interval > 0 else 0
            cv2.putText(
                img,
                f"PAD FPS: {send_fps:.1f}",
                (x, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        except Exception as e:
            print(f"Error in draw_controller: {e}")

    @lru_cache(maxsize=1)
    def get_labels(self):
        """ラベルファイルの読み込み（キャッシュ付き）"""
        try:
            with open(self.args.labels, "r") as f:
                labels = f.read().split("\n")
            return labels
        except Exception as e:
            print(f"Error loading labels: {e}")
            return ["Unknown"] * 10

    @staticmethod
    def get_camera_id(camera_num):
        """カメラ番号からカメラIDを検索"""
        try:
            cameras = Picamera2.global_camera_info()
            
            if 0 <= camera_num < len(cameras):
                return cameras[camera_num].get("Id", None)
            else:
                return None
        except Exception as e:
            print(f"Error getting camera ID: {e}")
            return None

    def run(self):
        """メインループ"""
        try:
            print("AI Controller running. Press Ctrl+C to exit.")
            print(f"コントローラー状態送信: 60FPS ({self.send_interval*1000:.2f}ms間隔)")
            
            while True:
                try:
                    # フレーム処理
                    self.process_frame()
                    
                    # 短いスリープでCPU使用率を下げつつ、60FPSに近づける
                    # 正確なタイミングはprocess_frame内で管理
                    time.sleep(0.001)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in main loop: {e}")
                    # 致命的でないエラーは続行
                    time.sleep(0.1)
        finally:
            # クリーンアップ
            if self.picam2:
                self.picam2.stop()
            if self.picam2_cam2:
                self.picam2_cam2.stop()
            print("AI Controller stopped.")


class Detection:
    """検出情報を保持するクラス"""
    def __init__(self, id, category, score, box):
        self.id = id
        self.category = category
        self.score = score
        self.box = box


def get_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="AI-powered Famicom controller")
    parser.add_argument("--preview", type=bool, default=True, help="show preview")
    parser.add_argument("--dual-camera", type=bool, default=True, help="Use dual camera")
    parser.add_argument("--model", type=str, default="./model/network.rpk", help="Path of the model")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second (reduced for stability)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection threshold")
    parser.add_argument("--capture", type=bool, default=True, help="Capture images")    
    parser.add_argument(
        "--labels",
        type=str,
        default="./model/labels.txt",
        help="Path to the labels file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = get_args()
        controller = FamicomAIController(args)
        controller.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)