import argparse
import time
import sys
from functools import lru_cache

import cv2
import numpy as np
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

from famicom_controller import FamicomControllerState, DebouncedFamicomControllerState, FamicomControllerSender


CAMERA_PREVIRE = True

# Mappings corresponding to category definitions
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
last_detections_cam2 = []
last_state = None

frame_count = 0
start_time = time.time()
fps = 0

class Detection:
    def __init__(self, id, category, score, box):
        self.id = id
        self.category = category
        self.score = score
        self.box =  box #imx500.convert_inference_coords(coords, metadata, picam2)

# Preview
def pre_callback(request):
    global last_state

    # draw information
    if CAMERA_PREVIRE:
        draw_detections(request)

    state = debounced_state.get_debounced_state()

    # Do not send if data is the same as before
    if not state.is_equal_to(last_state):
        fc_sender.send_command(state)
        last_state = state


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
    
    #debounced_state.update(raw_state)
    return raw_state

def parse_detections(imx500:IMX500, picam2:Picamera2):
    """
    Extract inference information from camera information
    """
    global frame_count, start_time, fps

    metadata = picam2.capture_metadata()

    # threshold
    threshold = args.threshold

    # Get detection results
    np_outputs = imx500.get_outputs(metadata, add_batch=True)

    # If no inference is involved
    if np_outputs is None: 
        return None

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        start_time = time.time()
        frame_count = 0

    # Sorting detection information
    boxes, scores, classes = np_outputs[0][0], np_outputs[2][0], np_outputs[1][0]

    # Set detection information above the default value
    detections = [
        Detection(picam2.camera_idx, category, score, imx500.convert_inference_coords(box, metadata, picam2))
        for box, score, category in zip(boxes, scores, classes)
        if score >= (threshold + 0.1 if category in {0, 1} else threshold)
    ]
    return detections

@lru_cache
def get_labels():
    with open(args.labels, "r") as f:
        labels = f.read().split("\n")

    return labels

def draw_detections(request, stream="main"):
    
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

        #cam1 detection
        draw_detection(m, last_detections)

        #cam1 detection
        if args.dual_camera:
            draw_detection(m, last_detections_cam2)



def draw_detection(img, detections):
    
    # Get book x Shize
    labels = get_labels()

    for detection in detections:
            
        # Get box size
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
        cv2.rectangle(img.array, (x, y), (x + w, y + h), color,2)

def draw_controller(img, x, y):
    """
    Draw the controller on the screen
    """
    
    # background rectangle
    cv2.rectangle(img, (x + 0, y + 0), (x + 151, y + 67), (160, 50, 50), -1)

    # central shape
    pts = np.array([[x + 7, y + 7], [x + 64, y + 7], [x + 64, y + 27], 
                    [x + 144, y + 27], [x +144, y + 60], [x + 7, y + 60]], np.int32)
    cv2.fillPoly(img, [pts], (230, 220, 190))

    cv2.line(img, (x+7, y + 48), (x + 144, y + 48), (0, 0, 0), 1)
    cv2.line(img, (x+7, y + 52), (x +144, y + 52), (0, 0, 0), 1)


    # pad
    cv2.rectangle(img, (x + 26, y + 27), (x + 36, y +  54), (0, 0, 0), -1)
    cv2.rectangle(img, (x + 17, y + 36), (x + 44, y + 46), (0, 0, 0), -1)

    # center button
    cv2.rectangle(img, (x + 52, y + 40), (x + 92, y + 54), (160, 50, 50), -1)
    cv2.rectangle(img, (x + 56, y + 45), (x + 67, y + 50), (0, 0, 0), -1)
    cv2.rectangle(img, (x + 77, y + 45), (x + 88, y + 50), (0, 0, 0), -1)

    # a / b
    cv2.circle(img, (x + 109, y + 47), 7, (0, 0, 0), -1)
    cv2.circle(img, (x + 127, y + 47), 7, (0, 0, 0), -1)

    # push 
    # up
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

# Check and get argument information
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=bool, default=True, help="show preview")
    parser.add_argument("--dual-camera", type=bool, default=False, help="Use dual camera")
    parser.add_argument("--model", type=str, default="./model/network.rpk", help="Path of the model")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--threshold", type=float, default=0.4, help="Detection threshold")
    parser.add_argument(
        "--labels",
        type=str,
        default="./model/labels.txt",
        help="Path to the labels file",
    )
    return parser.parse_args()

def get_camera_id(camera_num):
    """
    Find the camera ID from the camera number
    """
        
    cameras = Picamera2.global_camera_info()
    
    if 0 <= camera_num < len(cameras):
        return cameras[camera_num].get("Id", None)
    else:
        return None
    
if __name__ == "__main__":
    args = get_args()

    CAMERA_PREVIRE = args.preview

    # UART
    fc_sender = FamicomControllerSender('/dev/ttyAMA0')
    
    button_thresholds = {'A': (2,2), 'B':(2,2), 'RIGHT':(2,2)}

    debounced_state_cam1 = DebouncedFamicomControllerState(default_threshold_true=3, default_threshold_false=3, button_thresholds=button_thresholds)
    debounced_state_cam2 = DebouncedFamicomControllerState(default_threshold_true=3, default_threshold_false=3, button_thresholds=button_thresholds)
    debounced_state = DebouncedFamicomControllerState(default_threshold_true=3, default_threshold_false=3, button_thresholds=button_thresholds)
    
    # This must be called before instantiation of Picamera2
    # https://github.com/raspberrypi/picamera2/blob/main/picamera2/devices/imx500/imx500.py
    camera_id = get_camera_id(0)
    if camera_id == None:
        print("Error: AI Camera not found.")
        sys.exit(1)
    
    imx500 = IMX500(args.model, camera_id)

    # https://github.com/raspberrypi/picamera2
    # https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
    picam2 = Picamera2(0)

    # Bar display when model is installed
    imx500.show_network_fw_progress_bar()

    imx500_cam2 = None
    if args.dual_camera:
        camera_id = get_camera_id(1)
        if camera_id == None:
            print("Error: Second AI Camera2 not found.")
            sys.exit(1)     
        imx500_cam2 = IMX500(args.model, camera_id)
        imx500_cam2.show_network_fw_progress_bar()

    config = picam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=12)
    picam2.start(config, show_preview=CAMERA_PREVIRE)

    picam2_cam2 = None
    if args.dual_camera:
        picam2_cam2 = Picamera2(1)        
        config = picam2_cam2.create_preview_configuration(controls={"FrameRate": args.fps}, buffer_count=12)
        picam2_cam2.start(config, show_preview=False)

    # callback
    picam2.pre_callback = pre_callback

    #last_state = None

    while True:
        detections = parse_detections(imx500, picam2)
        if detections is not None:
            raw_state_cam1 = parse_state(detections)
            debounced_state_cam1.update(raw_state_cam1)
            last_detections = detections
    
        if args.dual_camera:
            detections_cam2 = parse_detections(imx500_cam2, picam2_cam2)
            if detections_cam2 is not None:
                raw_state_cam2 = parse_state(detections_cam2)
                debounced_state_cam2.update(raw_state_cam2)                
                last_detections_cam2 = detections_cam2

        debounced_state.update_consecutive_sum(debounced_state_cam1, debounced_state_cam2)
        # state = debounced_state.get_debounced_state()

        # # Do not send if data is the same as before
        # if not state.is_equal_to(last_state):
        #     fc_sender.send_command(state)
        #     last_state = state
