## what this ?

Docs [wiki.sipeed.com/m3axpi](https://wiki.sipeed.com/m3axpi)

## how to use

Based on the 20221125 of the Debian11 system.

- `pip3 install ax_pipeline_api -U`

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

### change ai model

```python
    pipeline.load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-m', b'/home/models/yolov5s_face_nv12_11.joint',
        b'-p', b'/home/config/yolov5s_face.json',
        b'-c', b'0',
    ])
```

## run demo code

### yolov5s

https://user-images.githubusercontent.com/32978053/204093040-179e35d0-8bfa-4626-b4cc-46f3f148eb71.mp4

```python
import pipeline
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
import pipeline
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
