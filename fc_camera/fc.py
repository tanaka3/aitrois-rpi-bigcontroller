import argparse
import time
from functools import lru_cache

import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500
import serial
from command_parser import CommandParser

ser = None
parser = None
last_detections = []

frame_count = 0
start_time = time.time()
fps = 0

# category の定義に対応するマッピング
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

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

# Preview
def pre_callback(request):
    # メタデータを整理する
    detections = parse_detections(request.get_metadata())

    # 情報を描画する
    draw_detections(request, detections)

    # データを送信する
    send_command(detections)


def send_command(detections):
    global ser, parser

    parser.reset()

    for detection in detections:
        if detection.category in CATEGORY_MAPPING:
            for attr in CATEGORY_MAPPING[detection.category]:
                if hasattr(parser, attr):
                    setattr(parser, attr, True)                    
    parser.send_command(ser)

# detectionを解析する
def parse_detections(metadata: dict):
    global last_detections, frame_count, start_time, fps

    # 閾値（引数で渡してる）
    threshold = args.threshold

    # 検出結果を取得
    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    # 画像サイズ
    input_w, input_h = imx500.get_input_size()

    # 見つからなかった場合
    if np_outputs is None:
        return last_detections

    # フレームをカウント
    frame_count += 1
    
    # 経過時間を計算
    elapsed_time = time.time() - start_time
    
    # 1秒経過したらFPSを計算
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        #print(f"FPS: {fps:.2f}")
        
        # 経過時間とフレーム数をリセット
        start_time = time.time()
        frame_count = 0

    # 検出情報を仕分ける
    boxes, scores, classes = np_outputs[0][0], np_outputs[2][0], np_outputs[1][0]

    # 規定値以上の検出情報を設定する
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes)
        if score >= (threshold - 0.1 if category in {0, 1} else threshold)
    ]
    return last_detections

# 結果がキャッシュされる
@lru_cache
def get_labels():
    with open(args.labels, "r") as f:
        labels = f.read().split("\n")

    return labels

def draw_detections(request, detections, stream="main"):
    
    # ラベルの情報を取得する（ファイルより）
    labels = get_labels()
    
    with MappedArray(request, stream) as m:

        fps_label = f"FPS:{fps:.2f}"
        cv2.putText(
            m.array,
            fps_label,
            (40, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )     

        for detection in detections:
            
            # ボックスのサイズを取得する
            x, y, w, h = detection.box

            color_map = {
                0: (255, 0, 0),       # 赤
                1: (0, 255, 0),       # 緑
                2: (0, 0, 255),       # 青
                3: (255, 165, 0),     # オレンジ
                4: (255, 223, 0),     # 黄
                5: (0, 255, 255),     # シアン
                6: (255, 0, 255),     # マゼンタ(
                7: (128, 128, 128),   # グレ
            }

            color = color_map.get(int(detection.category), (255, 255, 255))  # defaultは白色

            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"
            cv2.putText(
                m.array,
                label,
                (x + 5, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            cv2.rectangle(m.array, (x, y), (x + w, y + h), color,2)
       

# 引数の情報確認と、取得
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./model/network.rpk", help="Path of the model")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--threshold", type=float, default=0.50, help="Detection threshold")
    parser.add_argument(
        "--labels",
        type=str,
        default="./model/labels.txt",
        help="Path to the labels file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # UART
    ser = serial.Serial('/dev/ttyAMA0', baudrate=115200, timeout=1)
    parser = CommandParser()

    # This must be called before instantiation of Picamera2
    # https://github.com/raspberrypi/picamera2/blob/main/picamera2/devices/imx500/imx500.py
    imx500 = IMX500(args.model)

    # https://github.com/raspberrypi/picamera2
    # https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
    picam2 = Picamera2()

    # モデルインストール時のバー表示
    imx500.show_network_fw_progress_bar()

    # preview画像を表示するのに適した設定を生成する
    config = picam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=28)
    picam2.start(config, show_preview=True)

    # Preview用callback
    picam2.pre_callback = pre_callback

    while True:
        time.sleep(0.5)
