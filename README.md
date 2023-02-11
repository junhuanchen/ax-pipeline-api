## what this ?

[![MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![PyPI version](https://badge.fury.io/py/ax-pipeline-api.svg)](https://badge.fury.io/py/ax-pipeline-api)

This project is a Python implementation of [ax-pipeline](https://github.com/AXERA-TECH/ax-pipeline).

- `pip3 install ax-pipeline-api -U`

Based on AX620A Debian11 system. Docs at [wiki.sipeed.com/m3axpi](https://wiki.sipeed.com/m3axpi)

## new code (1.0.8)

>  It need update debian11(202202+) for pillow ImageFont(freetype).



```python

import m3axpi

from PIL import Image, ImageDraw, ImageFont

lcd_width, lcd_height, lcd_channel = 854, 480, 4

fnt = ImageFont.truetype("/home/res/sans.ttf", 20)
img = Image.new('RGBA', (lcd_width, lcd_height), (255,0,0,200))
ui = ImageDraw.ImageDraw(img)
ui.rectangle((20, 20, lcd_width-20, lcd_height-20), fill=(0,0,0,0), outline=(0,0,255,100), width=20)

logo = Image.open("/home/res/logo.png")
img.paste(logo, box=(lcd_width-logo.size[0], lcd_height-logo.size[1]), mask=None)

while True:
    rgba = img.copy()

    tmp = m3axpi.capture()
    rgb = Image.frombuffer("RGB", (tmp[1], tmp[0]), tmp[3])
    rgba.paste(rgb, box=(0, 0), mask=None) ## camera 320x180 paste 854x480

    res = m3axpi.forward()
    if 'nObjSize' in res:
        ui = ImageDraw.ImageDraw(rgba)
        ui.text((0, 0), "fps:%02d" % (res['niFps']), font=fnt)
        for obj in res['mObjects']:
            x, y, w, h = int(obj['bbox'][0]*lcd_width), int(obj['bbox'][1]*lcd_height), int(obj['bbox'][2]*lcd_width), int(obj['bbox'][3]*lcd_height)
            ui.rectangle((x,y,x+w,y+h), fill=(255,0,0,100), outline=(255,0,0,255))
            ui.text((x, y), "%s:%02d" % (obj['objname'], obj['prob']*100), font=fnt)
            rgba.paste(logo, box=(x+w-logo.size[1], y+h-logo.size[1]), mask=None)

    m3axpi.display([lcd_height, lcd_width, lcd_channel, rgba.tobytes()])

```

### test code

- [tests/test_m3axpi.py](tests/test_m3axpi.py)

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
        #     pipeline.drop()
pipeline.drop()

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
        #     pipeline.drop()
pipeline.drop()

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
- python3 setup.py build && cp build/lib*/m3axpi.*.so m3axpi.so
- rm -rf build && pip3 uninstall ax-pipeline-api -y && python3 setup.py build && pip3 install .
> pip3 install twine
- twine upload dist/* --verbose
