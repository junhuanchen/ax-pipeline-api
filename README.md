## what this ?

[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/ax-pipeline-api.svg)](https://badge.fury.io/py/ax-pipeline-api)

This project is a Python implementation of [ax-pipeline](https://github.com/AXERA-TECH/ax-pipeline).

- `pip3 install ax-pipeline-api -U`

Based on AX620A Debian11 system. Docs at [wiki.sipeed.com/m3axpi](https://wiki.sipeed.com/m3axpi)

## run code

### yolov5s

https://user-images.githubusercontent.com/32978053/204093040-179e35d0-8bfa-4626-b4cc-46f3f148eb71.mp4

```python
import time
from ax import pipeline

pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/config/yolov5s.json',
    '-c', '2',
])

while pipeline.work():
    time.sleep(0.001)
    tmp = pipeline.result()
    if tmp and tmp['nObjSize']:
        for i in tmp['mObjects']:
            x, y, w, h = i['bbox']['x'], i['bbox']['y'], i['bbox']['w'], i['bbox']['h']
            objname, objprob = i['objname'], i['prob']
            print(objname, objprob, x, y, w, h)
        # if tmp['nObjSize'] > 10: # try exit
        #     pipeline.free()
pipeline.free()

```

### yolov5s_face

https://user-images.githubusercontent.com/32978053/204444795-635299e9-89f1-4c76-9536-1c6dd9915b72.mp4

```python
import time
from ax import pipeline

pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/config/yolov5s_face.json',
    '-c', '2',
])

while pipeline.work():
    time.sleep(0.001)
    tmp = pipeline.result()
    if tmp and tmp['nObjSize']:
        for i in tmp['mObjects']:
            print(i)
        # if tmp['nObjSize'] > 10: # try exit
        #     pipeline.free()
pipeline.free()

```

### other demo

- [tests/test_yolov5s_pillow.py](tests/test_yolov5s_pillow.py)

https://user-images.githubusercontent.com/32978053/204150061-779c2443-5416-4a5e-a10f-e684a035510d.mp4

- [tests/test_ax_pose_print.py](tests/test_ax_pose_print.py)

https://user-images.githubusercontent.com/32978053/204150065-6977de65-423a-4895-a970-59ef914f9184.mp4

## more ?

### change sensor

- camera os04a10 is `'-c', '0',` and gc4653 is `'-c', '2',`.

```python
    pipeline.load([
        'libsample_vin_ivps_joint_vo_sipy.so',
        '-p', '/home/config/ax_pose.json',
        '-c', '0',
    ])
```

### change libxxx*.so

```python
    pipeline.load([
        'libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so',
        '-p', '/home/config/yolov5_seg.json',
        '-c', '0',
    ])
```

- Package with many inputs and outputs

```bash
libsample_h264_ivps_joint_vo_sipy.so            # input h264 video to ivps joint output screen vo
libsample_v4l2_user_ivps_joint_vo_sipy.so       # input v4l2 /dev/videoX to ivps joint output screen vo
libsample_rtsp_ivps_joint_rtsp_vo_sipy.so       # input video from rtsp to ivps joint output rtsp and screen vo
libsample_vin_ivps_joint_vo_sipy.so             # input mipi sensor to ivps joint output screen vo
libsample_vin_ivps_joint_venc_rtsp_sipy.so      # input mipi sensor to ivps joint output rtsp
libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so   # input mipi sensor to ivps joint output rtsp and screen vo
libsample_vin_ivps_joint_vo_h265_sipy.so        # input mipi sensor to ivps joint output screen vo and save h265 video file
```

### change ai model

```python
pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/config/yolov5s_face.json',
    '-c', '0',
])
```

- In-system models on /home/config.

```bash
ax_bvc_det.json		    hrnet_pose_yolov8.json	    yolov5s_face_recognition.json
ax_person_det.json	    license_plate_recognition.json  yolov5s_license_plate.json
ax_pose.json		    nanodet.json		    yolov6.json
ax_pose_yolov5s.json	    palm_hand_detection.json	    yolov7.json
ax_pose_yolov8.json	    pp_human_seg.json		    yolov7_face.json
crowdcount.json		    scrfd.json			    yolov7_palm_hand.json
hand_pose.json		    yolo_fastbody.json		    yolov8.json
hand_pose_yolov7_palm.json  yolopv2.json		    yolov8_seg.json
hrnet_animal_pose.json	    yolov5_seg.json		    yolox.json
hrnet_pose.json		    yolov5s.json
hrnet_pose_ax_det.json	    yolov5s_face.json
```

### pypi

- python3 setup.py sdist
- python3 setup.py build && pip3 install .
> pip3 install twine
- twine upload dist/* --verbose
