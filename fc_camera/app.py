import argparse
import time
from functools import lru_cache

import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

from famicom_controller import FamicomControllerState, DebouncedFamicomControllerState, FamicomControllerSender


CAMERA_PREVIRE = True

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

fc_sender = None
debounced_state = None
last_detections = []

frame_count = 0
start_time = time.time()
fps = 0

class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

# Preview
def pre_callback(request):

    # メタデータを整理する
    detections = parse_detections(request.get_metadata())

    if detections is not None:
        parse_state(detections)

    # 情報を描画する
    if CAMERA_PREVIRE:
        draw_detections(request, last_detections)

    if detections is not None:
        send_command()

def parse_state(detections):
    """
    Analyzes detections and converts them into controller information
    return: 
        FamicomControllerState: Controller status
    """
    raw_state = FamicomControllerState()

    for detection in detections:
        if detection.category in CATEGORY_MAPPING:
            for attr in CATEGORY_MAPPING[detection.category]:
                if hasattr(raw_state, attr):
                    setattr(raw_state, attr, True)
    
    debounced_state.update(raw_state)
    print(f"{raw_state.LEFT} -> {debounced_state.get_debounced_state().LEFT}")

def send_command():
    """
    Send information to the controller device    
    """
    #state.print_status()
    fc_sender.send_command(debounced_state.get_debounced_state())


def parse_detections(metadata: dict):
    """
    Extract inference information from camera information
    """

    global last_detections, frame_count, start_time, fps

    # 閾値（引数で渡してる）
    threshold = args.threshold

    # 検出結果を取得
    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    # 画像サイズ
    # input_w, input_h = imx500.get_input_size()

    # 見つからなかった場合
    if np_outputs is None: 
        return None

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

@lru_cache
def get_labels():
    with open(args.labels, "r") as f:
        labels = f.read().split("\n")

    return labels

def draw_detections(request, detections, stream="main"):
    
    # ラベルの情報を取得する（ファイルより）
    labels = get_labels()
    
    # withは、このステートから抜けるとメモリ開放してくれる便利な奴
    with MappedArray(request, stream) as m:

        # fps
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

        draw_controller(m.array, 450, 20) 

        for detection in detections:
            
            # ボックスのサイズを取得する
            x, y, w, h = detection.box

            color_map = {
                0: (255, 0, 0),       # 赤 (Aボタン) (RGB)
                1: (0, 255, 0),       # 緑 (Bボタン) (RGB)
                2: (0, 0, 255),       # 青 (下方向) (RGB)
                3: (255, 165, 0),     # オレンジ (左下方向) (RGB)
                4: (255, 223, 0),     # 黄 (右下方向) (RGB)
                5: (0, 255, 255),     # シアン (左方向) (RGB)
                6: (255, 0, 255),     # マゼンタ (右方向) (RGB)
                7: (128, 128, 128),   # グレー (選択ボタン) (RGB)
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
       

def draw_controller(img, x, y):

    
    # ベージュのエリア
    # 背景の矩形
    cv2.rectangle(img, (x + 0, y + 0), (x + 151, y + 67), (160, 50, 50), -1)

    # 中央の形状
    pts = np.array([[x + 7, y + 7], [x + 64, y + 7], [x + 64, y + 27], 
                    [x + 144, y + 27], [x +144, y + 60], [x + 7, y + 60]], np.int32)
    cv2.fillPoly(img, [pts], (230, 220, 190))

    cv2.line(img, (x+7, y + 48), (x + 144, y + 48), (0, 0, 0), 1)
    cv2.line(img, (x+7, y + 52), (x +144, y + 52), (0, 0, 0), 1)


    # 十字キー
    cv2.rectangle(img, (x + 26, y + 27), (x + 36, y +  54), (0, 0, 0), -1)
    cv2.rectangle(img, (x + 17, y + 36), (x + 44, y + 46), (0, 0, 0), -1)

    # 中央ボタン
    cv2.rectangle(img, (x + 52, y + 40), (x + 92, y + 54), (160, 50, 50), -1)
    cv2.rectangle(img, (x + 56, y + 45), (x + 67, y + 50), (0, 0, 0), -1)
    cv2.rectangle(img, (x + 77, y + 45), (x + 88, y + 50), (0, 0, 0), -1)

    # A/Bボタン
    cv2.circle(img, (x + 109, y + 47), 7, (0, 0, 0), -1)
    cv2.circle(img, (x + 127, y + 47), 7, (0, 0, 0), -1)

    # # push 
    # # up
    last_state = debounced_state.get_debounced_state()
    if last_state.UP:
        cv2.rectangle(img, (x + 28,  y + 29), (x + 34,  y + 37), (255, 0, 0), -1)
    # down
    if last_state.DOWN:    
        cv2.rectangle(img, (x + 28,  y + 44), (x + 34,  y + 52), (255, 0, 0), -1)
    
    # left
    if last_state.LEFT:
        cv2.rectangle(img, (x + 19,  y + 38), (x + 27,  y + 44), (255, 0, 0), -1)
    
    # right
    if last_state.RIGHT:
        cv2.rectangle(img, (x + 35,  y + 38), (x + 42,  y + 44), (255, 0, 0), -1)

    # select
    if last_state.SELECT:    
        cv2.rectangle(img, (x + 58,  y + 47), (x + 65,  y + 48), (255, 0, 0), -1)
    
    # start
    if last_state.START:    
        cv2.rectangle(img, (x + 79,  y + 47), (x + 86,  y + 48), (255, 0, 0), -1)
    
    # B
    if last_state.B:    
        cv2.circle(img, (x + 109,  y + 47), 5, (255, 0, 0), -1)
    # A
    if last_state.A:    
        cv2.circle(img, (x + 127,  y + 47), 5, (255, 0, 0), -1)

# 引数の情報確認と、取得
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=bool, default=True, help="show preview")
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

    CAMERA_PREVIRE = args.preview

    # UART
    #ser = serial.Serial('/dev/ttyAMA0', baudrate=115200, timeout=1)
    fc_sender = FamicomControllerSender('/dev/ttyAMA0')
    
    button_thresholds = {'A': (2,2), 'B':(2,2), 'RIGHT':(3,2)}
    debounced_state = DebouncedFamicomControllerState(default_threshold_true=3, default_threshold_false=3, button_thresholds=button_thresholds)
    
    # This must be called before instantiation of Picamera2
    # https://github.com/raspberrypi/picamera2/blob/main/picamera2/devices/imx500/imx500.py
    imx500 = IMX500(args.model)

    # https://github.com/raspberrypi/picamera2
    # https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
    picam2 = Picamera2()

    # モデルインストール時のバー表示
    imx500.show_network_fw_progress_bar()

    # preview画像を表示するのに適した設定を生成する
    # サイズ変えたいなら、main={"size":(800,600)}
    config = picam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=28)
    picam2.start(config, show_preview=CAMERA_PREVIRE)

    # Preview用callback
    picam2.pre_callback = pre_callback

    while True:
        time.sleep(0.5)
