
version='1.0.1'

import os
import ctypes
import collections
import time

_source = {
    "lib" : None,           # so 
    "path" : None,          # so path
    "config" : None,        # so config
    "queue" : None,         # result queue
    "work" : False,         # True is free
    "hide" : False,         # show pipeline draw result
    "camera" : False,       # allow camera ai input for debug
    "ai_image" : None,      # camera ai image(input)
    "display" : True,       # allow display for user_ui
    "ui_image" : None,      # display a image(display)
}

def config(key, value=None):
    if value != None:
        _source[key] = value
        if _source["camera"] is False:
            _source["ai_image"] = None
        if _source["display"] is False:
            _source["ui_image"] = None
        # print("dls", key, _source[key])
    return _source[key]

class _image:
    def __init__(self, width, height, mode, data):
        self.width = width
        self.height = height
        self.mode = mode
        self.data = data

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
    
'''
typedef enum _AX_NPU_CV_FrameDataType {
    AX_NPU_CV_FDT_UNKNOWN = 0,
    AX_NPU_CV_FDT_RAW10 = 1,
    AX_NPU_CV_FDT_RAW12 = 2,
    AX_NPU_CV_FDT_RAW16 = 3,
    AX_NPU_CV_FDT_NV12 = 4,
    AX_NPU_CV_FDT_NV21 = 5,
    AX_NPU_CV_FDT_RGB = 6,
    AX_NPU_CV_FDT_BGR = 7,
    AX_NPU_CV_FDT_RGBA = 8,
    AX_NPU_CV_FDT_GRAY = 9,
    AX_NPU_CV_FDT_YUV444 = 10,
    AX_NPU_CV_FDT_UV = 11,
    AX_NPU_CV_FDT_YUV422 = 12,
    AX_NPU_CV_FDT_BAYER_RGGB = 13,
    AX_NPU_CV_FDT_BAYER_GBRG = 14,
    AX_NPU_CV_FDT_BAYER_GRBG = 15,
    AX_NPU_CV_FDT_BAYER_BGGR = 16,
    AX_NPU_CV_FDT_UYVY = 17,
    AX_NPU_CV_FDT_YUYV = 18,
    AX_NPU_CV_FDT_YUV420_LEGACY = 19,
    AX_NPU_CV_FDT_LAB = 20,
} AX_NPU_CV_FrameDataType;
'''

class AX_NPU_CV_Stride(ctypes.Structure):
    _fields_ = [
        ("nH", ctypes.c_int),
        ("nW", ctypes.c_int),
        ("nC", ctypes.c_int),
    ]
    
class AX_NPU_CV_Image(ctypes.Structure):
    _fields_ = [
        ("pVir", ctypes.POINTER(ctypes.c_char)),
        ("pPhy", ctypes.c_int64),
        ("nSize", ctypes.c_int),
        ("nWidth", ctypes.c_int),
        ("nHeight", ctypes.c_int),        
        ("eDtype", ctypes.c_int),
        ("tStride", AX_NPU_CV_Stride),
    ]
    
def _result_callback(frame, result):
    res = ctypes.cast(result, ctypes.POINTER(sample_run_joint_results)).contents
    data = {}
    if res.nObjSize:
        data["nObjSize"] = res.nObjSize
        data["mObjects"] = []
        for i in range(res.nObjSize):
            obj = {}
            obj["label"] = res.mObjects[i].label
            obj["prob"] = res.mObjects[i].prob
            obj["objname"] = res.mObjects[i].objname
            obj["bbox"] = {
                "x" : res.mObjects[i].bbox.x,
                "y" : res.mObjects[i].bbox.y,
                "w" : res.mObjects[i].bbox.w,
                "h" : res.mObjects[i].bbox.h,
            }
            if res.mObjects[i].bHasFaceLmk:
                obj["bHasFaceLmk"] = res.mObjects[i].bHasFaceLmk
                obj["face_landmark"] = []
                for j in range(5):
                    obj["face_landmark"].append({
                        "x" : res.mObjects[i].face_landmark[j].x,
                        "y" : res.mObjects[i].face_landmark[j].y,
                    })
            if res.mObjects[i].bHasPoseLmk:
                obj["bHasPoseLmk"] = res.mObjects[i].bHasPoseLmk
                obj["pose_landmark"] = []
                for j in range(17):
                    obj["pose_landmark"].append({
                        "x" : res.mObjects[i].pose_landmark[j].x,
                        "y" : res.mObjects[i].pose_landmark[j].y,
                    })
            if res.mObjects[i].bHaseMask:
                obj["bHaseMask"] = res.mObjects[i].bHaseMask
                obj["mYolov5Mask"] = {
                    "w" : res.mObjects[i].mYolov5Mask.w,
                    "h" : res.mObjects[i].mYolov5Mask.h,
                    "data" : ctypes.string_at(res.mObjects[i].mYolov5Mask.data, res.mObjects[i].mYolov5Mask.w*res.mObjects[i].mYolov5Mask.h),
                }
            data["mObjects"].append(obj)
    if res.bPPHumSeg:
        data["bPPHumSeg"] = res.bPPHumSeg
        data["mPPHumSeg"] = []
        for i in range(192*192):
            data["mPPHumSeg"].append(res.mPPHumSeg.mask[i])
        data["mPPHumSeg"] = bytes(data["mPPHumSeg"])
    if len(data):
        data['time'] = time.time()
        _source["queue"].append(data)
        # print(data)
    if _source["camera"]:
        img = ctypes.cast(frame, ctypes.POINTER(AX_NPU_CV_Image)).contents
        if img.eDtype == 7: # is rgb camera
            _source["ai_image"] = _image(img.nWidth, img.nHeight, "RGB", ctypes.string_at(img.pVir, img.nWidth * img.nHeight * 3))
        else:
            _source["camera"] = False
    return 0

def _display_callback(height, width, mode, data):
    if _source["display"]:
        img = _source["ui_image"]
        if img and height == img.height and width == img.width:
            # print(height, width, mode)
            # print(img.height, img.width, img.mode)
            tmp = ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))
            buf = bytearray(img.data)
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            ctypes.memmove(tmp.contents, ptr, len(buf))
            # _source["image"] = None
    return _source["hide"]

def work():
    return _source["work"]

def result():
    if len(_source["queue"]):
        return _source["queue"].popleft()
    return None

def free():
    if _source["work"] == True:
        _source["lib"].__sigExit.argtypes = [ctypes.c_int]
        _source["lib"].__sigExit.restype = None
        ret = _source["lib"].__sigExit(ctypes.c_int(0))

def load(config, maxsize=10):
    if _source["work"] == False:
        _source["config"] = config
        _source["queue"] = collections.deque(maxlen=maxsize)
        _source["path"] = os.path.join(os.path.dirname(__file__), "lib", str(config[0], encoding="iso-8859-1"))
        _source["lib"] = ctypes.CDLL(_source["path"])
        CB_RESULT = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(sample_run_joint_results),
        )
        cb_result = CB_RESULT(_result_callback)
        _source["lib"].register_result_callback.argtypes = [CB_RESULT]
        _source["lib"].register_result_callback.restype = ctypes.c_int
        ret = _source["lib"].register_result_callback(cb_result)
        CB_DISPLAY = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        )
        cb_display = CB_DISPLAY(_display_callback)
        _source["lib"].register_display_callback.argtypes = [CB_DISPLAY]
        _source["lib"].register_display_callback.restype = ctypes.c_int
        ret = _source["lib"].register_display_callback(cb_display)
        main_msg = (ctypes.c_char_p * len(config))()
        for i in range(len(config)):
            main_msg[i] = config[i]
        _source["lib"].main.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        _source["lib"].main.restype = None
        # print("main: ", _source)
        _source["work"] = True
        ret = _source["lib"].main(len(main_msg), main_msg)
        _source["queue"].clear()
        _source["lib"], _source["path"], _source["config"] = None, None, None
        _source["work"] = False

def unit_test_yolov5s():

    import threading
    def print_data(threadName, delay):
        print("print_data 1", threadName, work())
        # while work():
        for i in range(200):
            time.sleep(delay)
            tmp = result()
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
        
    ])

    test.join()

def unit_test_ax_pose():

    import threading
    def print_data(threadName, delay):
        print("print_data 1", threadName, work())
        # while work():
        for i in range(200):
            time.sleep(delay)
            tmp = result()
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
    ])

    test.join()

def unit_test_yolov5s_seg():
    import threading
    def print_data(threadName, delay):
        print("print_data 1", threadName, work())
        # while work():
        for i in range(200):
            time.sleep(delay)
            tmp = result()
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
    ])

    test.join()

def unit_test_display():
    try:
        lcd_width, lcd_height = 854, 480
        from PIL import Image, ImageDraw
        logo = Image.open("/home/res/logo.png")
        img = Image.new('RGBA', (lcd_width, lcd_height), (255,0,0,200))
        ui = ImageDraw.ImageDraw(img)
        ui.rectangle((54,40,800,400), fill=(0,0,0,0), outline=(0,0,255,100), width=100)
        img.paste(logo, box=(854//2, 480//2), mask=None)
        r,g,b,a = img.split()
        src_argb = Image.merge("RGBA", (a,b,g,r))
        config("ui_image", _image(854, 480, "ARGB", src_argb.tobytes()))
        import threading
        def print_data(threadName, delay):
            print("print_data 1", threadName, work())
            # while work():
            for i in range(60):
                time.sleep(delay)
                tmp = result()
                if tmp:
                    print(tmp)
            config("camera", True)
            for i in range(40):
                time.sleep(delay)
                tmp = result()
                if tmp:
                    print(tmp)
                ai = config("ai_image")
                if ai and ai.mode == "RGB":
                    tmp = Image.frombytes("RGB", (ai.width, ai.height), ai.data)
                    tmp.thumbnail((ai.width // 2, ai.height // 2))
                    img.paste(tmp, box=(0, 0), mask=None)
                    r,g,b,a = img.split()
                    argb = Image.merge("RGBA", (a,b,g,r)).tobytes()
                    config("ui_image", _image(lcd_width, lcd_height, "ARGB", argb))
            config("camera", False)
            config("hide", True)
            for i in range(200):
                time.sleep(delay)
                tmp = result()
                if tmp and tmp['nObjSize']:
                    src_argb = Image.merge("RGBA", (a,b,g,r))
                    ui = ImageDraw.ImageDraw(src_argb)
                    for i in tmp['mObjects']:
                        x = i['bbox']['x'] * lcd_width
                        y = i['bbox']['y'] * lcd_height
                        w = i['bbox']['w'] * lcd_width
                        h = i['bbox']['h'] * lcd_height
                        objname = i['objname']
                        objprob = i['prob']
                        ui.rectangle((x,y,x+w,y+h), fill=(100,0,0,255), outline=(255,0,0,255))
                        ui.text((x,y), objname, fill=(255,0,0,255))
                        ui.text((x,y+20), str(objprob), fill=(255,0,0,255))
                    config("ui_image", _image(854, 480, "ARGB", src_argb.tobytes()))
            config("hide", False)
            config("display", False)
            free()
            print("print_data 2", work())
        test = threading.Thread(target=print_data, args=("Thread-1", 0.05, ))
        test.start()

        load([
            b'libsample_vin_ivps_joint_vo_sipy.so',
            b'-m', b'/home/models/yolov5s.joint',
            b'-p', b'/home/config/yolov5s.json',
            b'-c', b'0',
        ])

        test.join()
    except Exception as e:
        print("apt install python3-pil -y")

if __name__ == "__main__":
    unit_test_display()
    unit_test_yolov5s()
    unit_test_ax_pose()
    unit_test_yolov5s_seg()
