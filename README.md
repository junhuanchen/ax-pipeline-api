## what this ?

[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/ax-pipeline-api.svg)](https://badge.fury.io/py/ax-pipeline-api)

Docs at [wiki.sipeed.com/m3axpi](https://wiki.sipeed.com/m3axpi)

## how to use

Based on the 20221125 of the Debian11 system.

- `pip3 install ax-pipline-api -U`

### change sensor

- camera os04a10 is `b'-c', b'0',` and gc4653 is `b'-c', b'2',`.

```python
    pipeline.load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-p', b'/home/config/ax_pose.json',
        b'-c', b'0',
    ])
```

### change libxxx*.so

```python
    pipeline.load([
        b'libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so',
        b'-m', b'/home/models/yolov5s-seg.joint',
        b'-p', b'/home/config/yolov5_seg.json',
        b'-c', b'0',
    ])
```

- Package with many inputs and outputs

```bash
libsample_h264_ivps_joint_vo_sipy.so            # input h264 video to ivps joint output screen vo
libsample_v4l2_user_ivps_joint_vo_sipy.so       # input v4l2 /dev/videoX to ivps joint output screen vo
libsample_rtsp_ivps_joint_sipy.so               # input video from rtsp to ivps joint
libsample_rtsp_ivps_joint_rtsp_vo_sipy.so       # input video from rtsp to ivps joint output rtsp and screen vo
libsample_vin_ivps_joint_vo_sipy.so             # input mipi sensor to ivps joint output screen vo
libsample_vin_ivps_joint_venc_rtsp_sipy.so      # input mipi sensor to ivps joint output rtsp
libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so   # input mipi sensor to ivps joint output rtsp and screen vo
libsample_vin_ivps_joint_vo_h265_sipy.so        # input mipi sensor to ivps joint output screen vo and save h265 video file.
libsample_vin_joint_sipy.so                     # input mipi sensor to ivps joint
```

### change ai model

```python
    pipeline.load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-m', b'/home/models/yolov5s_face_nv12_11.joint',
        b'-p', b'/home/config/yolov5s_face.json',
        b'-c', b'0',
    ])
```

- In-system models on /home/config.

```bash
ax_person_det.json          license_plate_recognition.json  yolov5s_face.json
ax_pose.json                nanodet.json                    yolov5s_face_recognition.json
ax_pose_yolov5s.json        palm_hand_detection.json        yolov5s_license_plate.json
hand_pose.json              pp_human_seg.json               yolov6.json
hand_pose_yolov7_palm.json  yolo_fastbody.json              yolov7.json
hrnet_animal_pose.json      yolopv2.json                    yolov7_face.json
hrnet_pose.json             yolov5_seg.json                 yolov7_palm_hand.json
hrnet_pose_ax_det.json      yolov5s.json                    yolox.json
```

## run demo code

### yolov5s

https://user-images.githubusercontent.com/32978053/204093040-179e35d0-8bfa-4626-b4cc-46f3f148eb71.mp4

```python
from ax import pipeline
import time
import threading

def pipeline_data(threadName, delay):
    time.sleep(0.2) # wait for pipeline.work() is True
    for i in range(400):
        time.sleep(delay)
        tmp = pipeline.result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                x = i['bbox']['x']
                y = i['bbox']['y']
                w = i['bbox']['w']
                h = i['bbox']['h']
                objname = i['objname']
                objprob = i['prob']
                print(objname, objprob)
    pipeline.free() # 400 * 0.05s auto exit pipeline

thread = threading.Thread(target=pipeline_data, args=("Thread-1", 0.05, ))
thread.start()

pipeline.load([
    b'libsample_vin_ivps_joint_vo_sipy.so',
    b'-m', b'/home/models/yolov5s.joint',
    b'-p', b'/home/config/yolov5s.json',
    b'-c', b'2',
])

thread.join() # wait thread exit
```

### yolov5s_face

https://user-images.githubusercontent.com/32978053/204444795-635299e9-89f1-4c76-9536-1c6dd9915b72.mp4


```python
from ax import pipeline
import time
import threading

def pipeline_data(threadName, delay):
    time.sleep(0.2) # wait for pipeline.work() is True
    for i in range(400):
        time.sleep(delay)
        tmp = pipeline.result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    pipeline.free() # 400 * 0.05s auto exit pipeline

thread = threading.Thread(target=pipeline_data, args=("Thread-1", 0.05, ))
thread.start()

pipeline.load([
    b'libsample_vin_ivps_joint_vo_sipy.so',
    b'-m', b'/home/models/yolov5s_face_nv12_11.joint',
    b'-p', b'/home/config/yolov5s_face.json',
    b'-c', b'0',
])

thread.join() # wait thread exit
```

## more example

- [tests/test_yolov5s_pillow.py](tests/test_yolov5s_pillow.py)

https://user-images.githubusercontent.com/32978053/204150061-779c2443-5416-4a5e-a10f-e684a035510d.mp4

- [tests/test_ax_pose_print.py](tests/test_ax_pose_print.py)

https://user-images.githubusercontent.com/32978053/204150065-6977de65-423a-4895-a970-59ef914f9184.mp4
