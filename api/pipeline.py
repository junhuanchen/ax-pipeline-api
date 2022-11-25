
import os
import ctypes
import collections
import time

source = {
    "lib" : None,
    "path" : None,
    "config" : None,
    "queue" : None,
    "work" : False,
}

class sample_run_joint_bbox(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
    ]

class sample_run_joint_point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
    ]

class sample_run_joint_mat(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("data", ctypes.c_char_p),
    ]

class sample_run_joint_object(ctypes.Structure):
    _fields_ = [
        ("bbox", sample_run_joint_bbox),
        ("bHasFaceLmk", ctypes.c_int),
        ("face_landmark", sample_run_joint_point*5),
        ("bHasPoseLmk", ctypes.c_int),
        ("pose_landmark", sample_run_joint_point*17),
        ("bHaseMask", ctypes.c_int),
        ("mYolov5Mask", sample_run_joint_mat),
        ("label", ctypes.c_int),
        ("prob", ctypes.c_float),
        ("objname", ctypes.c_char*16),
    ]

class sample_run_joint_pphumseg(ctypes.Structure):
    _fields_ = [
        ("mask", ctypes.c_byte*(192*192)),
    ]

class sample_run_joint_results(ctypes.Structure):
    _fields_ = [
        ("nObjSize", ctypes.c_int),
        ("mObjects", sample_run_joint_object*64),
        ("bPPHumSeg", ctypes.c_int),
        ("mPPHumSeg", sample_run_joint_pphumseg),
    ]

def __callback(size, result):
    tmp = ctypes.cast(result, ctypes.POINTER(sample_run_joint_results))
    data = {}
    if tmp.contents.nObjSize:
        data["nObjSize"] = tmp.contents.nObjSize
        data["mObjects"] = []
        for i in range(tmp.contents.nObjSize):
            obj = {}
            obj["label"] = tmp.contents.mObjects[i].label
            obj["prob"] = tmp.contents.mObjects[i].prob
            obj["objname"] = tmp.contents.mObjects[i].objname
            obj["bbox"] = {
                "x" : tmp.contents.mObjects[i].bbox.x,
                "y" : tmp.contents.mObjects[i].bbox.y,
                "w" : tmp.contents.mObjects[i].bbox.w,
                "h" : tmp.contents.mObjects[i].bbox.h,
            }
            if tmp.contents.mObjects[i].bHasFaceLmk:
                obj["bHasFaceLmk"] = tmp.contents.mObjects[i].bHasFaceLmk
                obj["face_landmark"] = []
                for j in range(5):
                    obj["face_landmark"].append({
                        "x" : tmp.contents.mObjects[i].face_landmark[j].x,
                        "y" : tmp.contents.mObjects[i].face_landmark[j].y,
                    })
            if tmp.contents.mObjects[i].bHasPoseLmk:
                obj["bHasPoseLmk"] = tmp.contents.mObjects[i].bHasPoseLmk
                obj["pose_landmark"] = []
                for j in range(17):
                    obj["pose_landmark"].append({
                        "x" : tmp.contents.mObjects[i].pose_landmark[j].x,
                        "y" : tmp.contents.mObjects[i].pose_landmark[j].y,
                    })
            if tmp.contents.mObjects[i].bHaseMask:
                obj["bHaseMask"] = tmp.contents.mObjects[i].bHaseMask
                obj["mYolov5Mask"] = {
                    "w" : tmp.contents.mObjects[i].mYolov5Mask.w,
                    "h" : tmp.contents.mObjects[i].mYolov5Mask.h,
                    "data" : ctypes.string_at(tmp.contents.mObjects[i].mYolov5Mask.data, tmp.contents.mObjects[i].mYolov5Mask.w*tmp.contents.mObjects[i].mYolov5Mask.h),
                }
            data["mObjects"].append(obj)
    if tmp.contents.bPPHumSeg:
        data["bPPHumSeg"] = tmp.contents.bPPHumSeg
        data["mPPHumSeg"] = []
        for i in range(192*192):
            data["mPPHumSeg"].append(tmp.contents.mPPHumSeg.mask[i])
        data["mPPHumSeg"] = bytes(data["mPPHumSeg"])
    if len(data):
        data['time'] = time.time()
        source["queue"].append(data)
        # print(data)
    return 0

def work():
    return source["work"]

def data():
    if len(source["queue"]):
        return source["queue"].popleft()
    return None

def free():
    if source["work"] == True:
        source["lib"].__sigExit.argtypes = [ctypes.c_int]
        source["lib"].__sigExit.restype = None
        ret = source["lib"].__sigExit(ctypes.c_int(0))

def load(config, maxsize=10):
    if source["work"] == False:
        source["work"] = True
        source["config"] = config
        source["queue"] = collections.deque(maxlen=maxsize)
        source["path"] = os.path.join(os.path.dirname(__file__), "lib", str(config[0], encoding="iso-8859-1"))
        source["lib"] = ctypes.CDLL(source["path"])
        CB_FUNC = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(sample_run_joint_results),
        )
        cb_func = CB_FUNC(__callback)
        source["lib"].register_result_callback.argtypes = [CB_FUNC]
        source["lib"].register_result_callback.restype = ctypes.c_int
        ret = source["lib"].register_result_callback(cb_func)
        main_msg = (ctypes.c_char_p * len(config))()
        for i in range(len(config)):
            main_msg[i] = config[i]
        source["lib"].main.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        source["lib"].main.restype = None
        print("main: ", source)
        ret = source["lib"].main(len(main_msg), main_msg)
        source["queue"].clear()
        source["lib"], source["path"], source["config"] = None, None, None
        source["work"] = False

def unit_test_yolov5s():

    import time, threading
    def print_data(threadName, delay):
        print("print_data 1", work())
        # while work():
        for i in range(200):
            time.sleep(delay)
            tmp = data()
            if tmp:
                print(tmp)
        free()
        print("print_data 2", work())
    test = threading.Thread(target=print_data, args=("Thread-1", 0.05, ))
    test.start()

    load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-m', b'/home/models/yolov5s.joint',
        b'-p', b'/home/config/yolov5s.json',
        b'-c', b'0',
        b'-e', b'2',
    ])

    test.join()

def unit_test_ax_pose():

    import time, threading
    def print_data(threadName, delay):
        print("print_data 1", work())
        # while work():
        for i in range(200):
            time.sleep(delay)
            tmp = data()
            if tmp:
                print(tmp)
        free()
        print("print_data 2", work())
    test = threading.Thread(target=print_data, args=("Thread-1", 0.05, ))
    test.start()

    load([
        b'libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so',
        b'-p', b'/home/config/ax_pose.json',
        b'-c', b'0',
        b'-e', b'2',
    ])

    test.join()

def unit_test_yolov5s_seg():

    import time, threading
    def print_data(threadName, delay):
        print("print_data 1", work())
        # while work():
        for i in range(200):
            time.sleep(delay)
            tmp = data()
            if tmp:
                print(tmp)
        free()
        print("print_data 2", work())
    test = threading.Thread(target=print_data, args=("Thread-1", 0.05, ))
    test.start()

    load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-m', b'/home/models/yolov5s-seg.joint',
        b'-p', b'/home/config/yolov5_seg.json',
        b'-c', b'0',
        b'-e', b'2',
    ])

    test.join()

if __name__ == "__main__":
    for i in range(2):
        unit_test_yolov5s()
        unit_test_ax_pose()
        unit_test_yolov5s_seg()
