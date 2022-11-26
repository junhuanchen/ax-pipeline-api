## what this ?

Docs [wiki.sipeed.com/m3axpi](https://wiki.sipeed.com/m3axpi)

## how to use

On the m3axpi board 20221125 the basis of the Debian11 system.

- `pip3 install ax_pipeline_api -U`

## run demo code

```python
import pipeline
import time
import threading
def print_data(threadName, delay):
    print("print_data 1", threadName, pipeline.work())
    # while work():
    for i in range(1000):
        time.sleep(delay)
        tmp = pipeline.result()
        if tmp:
            print(tmp)
    pipeline.free()
    print("print_data 2", pipeline.work())

test = threading.Thread(target=print_data, args=("Thread-1", 0.05, ))
test.start()

pipeline.load([
    b'libsample_vin_ivps_joint_vo_sipy.so',
    b'-m', b'/home/models/yolov5s.joint',
    b'-p', b'/home/config/yolov5s.json',
    b'-c', b'0',
])

test.join()
```

## more example

- `python3 tests/test_ax_pose_print.py`
- `python3 tests/test_yolov5s_pillow.py`

## change sensor

- camera os04a10 is `b'-c', b'0',` and gc4653 is `b'-c', b'2',`.

```python
    pipeline.load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-p', b'/home/config/ax_pose.json',
        b'-c', b'0',
    ])
```

## change libxxx*.so

```python
    pipeline.load([
        b'-m', b'/home/models/yolov5s-seg.joint',
        b'-p', b'/home/config/yolov5_seg.json',
        b'-c', b'0',
    ])
```

## change ai model

```python
    pipeline.load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-m', b'/home/models/yolov5s.joint',
        b'-p', b'/home/config/yolov5s.json',
        b'-c', b'0',
    ])
```
