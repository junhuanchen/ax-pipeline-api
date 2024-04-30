
version='1.1.3'

import os
import ctypes
import collections
import time
import threading

_source = {
    "lib" : None,           # so
    "path" : None,          # so path
    "config" : None,        # so config
    "queue" : None,         # result queue
    "thread" : None,        # thread for work
    "hide" : False,         # show pipeline draw result
    "input" : False,       # allow camera ai input for debug
    "camera" : None,      # camera ai image(input)
    "output" : True,       # allow display for user_ui
    "display" : None,      # display a image(display)
}

class pipeline_event(threading.Thread):
    def __init__(self, _source):
        threading.Thread.__init__(self)
        self._source = _source
    def run(self):
        config = self._source["config"]
        self._source["lib"] = ctypes.CDLL(self._source["path"])
        CB_RESULT = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(axdl_results_t),
        )
        cb_result = CB_RESULT(_result_callback)
        self._source["lib"].register_result_callback.argtypes = [CB_RESULT]
        self._source["lib"].register_result_callback.restype = ctypes.c_int
        ret = self._source["lib"].register_result_callback(cb_result)
        CB_DISPLAY = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        )
        cb_display = CB_DISPLAY(_display_callback)
        self._source["lib"].register_display_callback.argtypes = [CB_DISPLAY]
        self._source["lib"].register_display_callback.restype = ctypes.c_int
        ret = self._source["lib"].register_display_callback(cb_display)
        main_msg = (ctypes.c_char_p * len(config))()
        for i in range(len(config)):
            main_msg[i] = bytes(config[i], encoding="iso-8859-1")
        self._source["lib"].main.argtypes = ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
        self._source["lib"].main.restype = None
        ret = self._source["lib"].main(len(main_msg), main_msg)
        self._source["queue"].clear()
        self._source["lib"], self._source["path"], self._source["config"] = None, None, None
        self._source["thread"] = None
        self._source["hide"] = False
        self._source["input"] = False
        self._source["camera"] = None
        self._source["output"] = True
        self._source["display"] = None

class _image:
    def __init__(self, width, height, mode, data):
        self.width = width
        self.height = height
        self.mode = mode
        self.data = data

def config(key, value=None):
    if value != None:
        _source[key] = value
        if _source["input"] is False:
            _source["camera"] = None
        if _source["output"] is False:
            _source["display"] = None
        if key == "display":
            _source[key] = _image(value[0], value[1], value[2], value[3])
        # print("dls", key, _source[key])
    return _source[key]

class axdl_bbox_t(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("w", ctypes.c_float),
        ("h", ctypes.c_float),
    ]

class axdl_point_t(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("score", ctypes.c_float),
    ]

class axdl_mat_t(ctypes.Structure):
    _fields_ = [
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("c", ctypes.c_int),
        ("s", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]

class axdl_object_t(ctypes.Structure):
    _fields_ = [
        ("bbox", axdl_bbox_t),
        ("bHasBoxVertices", ctypes.c_int),
        ("bbox_vertices", axdl_point_t*4), # bbox with rotate
        ("nLandmark", ctypes.c_int), # none 0 face 5 body 17 animal 20 hand 21
        ("landmark", ctypes.POINTER(axdl_point_t)),
        ("bHasMask", ctypes.c_int),
        ("mYolov5Mask", axdl_mat_t),
        ("bHasFaceFeat", ctypes.c_int),
        ("mFaceFeat", axdl_mat_t),
        ("label", ctypes.c_int),
        ("prob", ctypes.c_float),
        ("track_id", ctypes.c_long),
        ("objname", ctypes.c_char*20),
    ]

class axdl_results_t(ctypes.Structure):
    _fields_ = [
        ("mModelType", ctypes.c_int),
        ("bObjTrack", ctypes.c_int),
        ("nObjSize", ctypes.c_int),
        ("mObjects", axdl_object_t*64),
        ("bPPHumSeg", ctypes.c_int),
        ("mPPHumSeg", axdl_mat_t),
        ("bYolopv2Mask", ctypes.c_int),
        ("mYolopv2seg", axdl_mat_t),
        ("mYolopv2ll", axdl_mat_t),
        ("nCrowdCount", ctypes.c_int),
        ("mCrowdCountPts", ctypes.POINTER(axdl_point_t)),
        ("niFps", ctypes.c_int),
        ("noFps", ctypes.c_int),
    ]

class axdl_image_t(ctypes.Structure):
    _fields_ = [
        ("pPhy", ctypes.c_ulonglong),
        ("pVir", ctypes.c_void_p),
        ("nSize", ctypes.c_uint),
        ("nWidth", ctypes.c_uint),
        ("nHeight", ctypes.c_uint),
        ("eDtype", ctypes.c_int),
        ("tStride_H", ctypes.c_int),
        ("tStride_W", ctypes.c_int),
        ("tStride_C", ctypes.c_int),
    ]

def _result_callback(frame, result):
    # print("result_callback", frame, result)
    res = ctypes.cast(result, ctypes.POINTER(axdl_results_t)).contents
    data = {}
    if res.nObjSize:
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
            if res.mObjects[i].bHasBoxVertices:
                obj["bHasBoxVertices"] = res.mObjects[i].bHasBoxVertices
                obj["bbox_vertices"] = []
                for j in range(4):
                    obj["bbox_vertices"].append({
                        "x" : res.mObjects[i].bbox_vertices[j].x,
                        "y" : res.mObjects[i].bbox_vertices[j].y,
                    })
            if res.mObjects[i].nLandmark:
                obj["nLandmark"] = res.mObjects[i].nLandmark
                obj["landmark"] = []
                for j in range(res.mObjects[i].nLandmark):
                    obj["landmark"].append({
                        "x" : res.mObjects[i].landmark[j].x,
                        "y" : res.mObjects[i].landmark[j].y,
                    })
            if res.mObjects[i].bHasMask:
                obj["bHasMask"] = res.mObjects[i].bHasMask
                obj["mYolov5Mask"] = {
                    "w" : res.mObjects[i].mYolov5Mask.w,
                    "h" : res.mObjects[i].mYolov5Mask.h,
                    "data" : ctypes.string_at(res.mObjects[i].mYolov5Mask.data, res.mObjects[i].mYolov5Mask.w*res.mObjects[i].mYolov5Mask.h),
                }
            if res.mObjects[i].bHasFaceFeat:
                obj["bHasFaceFeat"] = res.mObjects[i].bHasFaceFeat
                obj["mFaceFeat"] = {
                    "w" : res.mObjects[i].mFaceFeat.w,
                    "h" : res.mObjects[i].mFaceFeat.h,
                    "data" : ctypes.string_at(res.mObjects[i].mFaceFeat.data, res.mObjects[i].mFaceFeat.w*res.mObjects[i].mFaceFeat.h),
                }
            data["mObjects"].append(obj)
        data["nObjSize"] = res.nObjSize
    ## There is a problem taking out the mask data ##
    if res.bPPHumSeg:
        data["bPPHumSeg"] = res.bPPHumSeg
        data["mPPHumSeg"] = {
            "w" : res.mPPHumSeg.w,
            "h" : res.mPPHumSeg.h,
            "data" : ctypes.string_at(res.mPPHumSeg.data, res.mPPHumSeg.w*res.mPPHumSeg.h),
        }
    if res.bYolopv2Mask:
        data["bYolopv2Mask"] = res.bYolopv2Mask
        data["mYolopv2seg"] = {
            "w" : res.mYolopv2seg.w,
            "h" : res.mYolopv2seg.h,
            "data" : ctypes.string_at(res.mYolopv2seg.data, res.mYolopv2seg.w*res.mYolopv2seg.h),
        }
        data["mYolopv2ll"] = {
            "w" : res.mYolopv2ll.w,
            "h" : res.mYolopv2ll.h,
            "data" : ctypes.string_at(res.mYolopv2ll.data, res.mYolopv2ll.w*res.mYolopv2ll.h),
        }
    if res.nCrowdCount:
        data["nCrowdCount"] = res.nCrowdCount
        data["mCrowdCountPts"] = []
        for i in range(res.nCrowdCount):
            data["mCrowdCountPts"].append({
                "x" : res.mCrowdCountPts[i].x,
                "y" : res.mCrowdCountPts[i].y,
            })
    if len(data):
        data["mModelType"] = res.mModelType
        data['time'] = time.time()
        _source["queue"].append(data)
        # print(data)
    if _source["input"]:
        img = ctypes.cast(frame, ctypes.POINTER(axdl_image_t)).contents
        if img.eDtype > 2: # is axdl_color_space_bgr(3) or axdl_color_space_rgb(4)
            _source["camera"] = _image(img.nWidth, img.nHeight, "RGB", ctypes.string_at(img.pVir, img.nWidth * img.nHeight * 3))
        else:
            _source["input"] = False
    return 0

def _display_callback(height, width, mode, data):
    if _source["output"]:
        img = _source["display"]
        if isinstance(img, _image) and height == img.height and width == img.width:
            tmp = ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))
            buf = bytearray(img.data)
            ptr = (ctypes.c_char * len(buf)).from_buffer(buf)
            ctypes.memmove(tmp.contents, ptr, len(buf))
    return _source["hide"]

def work():
    return _source["thread"].is_alive() if _source["thread"] else False

def result():
    if _source["queue"] and len(_source["queue"]):
        return _source["queue"].popleft()
    return None

def drop():
    if _source["thread"]:
        _source["lib"].__sigExit.argtypes = [ctypes.c_int]
        _source["lib"].__sigExit.restype = None
        ret = _source["lib"].__sigExit(ctypes.c_int(0))
        _source["thread"].join()

def load(config, maxsize=10):
    if _source["thread"] == None:
        _source["config"] = config
        _source["queue"] = collections.deque(maxlen=maxsize)
        _source["path"] = os.path.join(os.path.dirname(__file__), "lib", config[0])
        # print("main: ", _source)
        _source["thread"] = pipeline_event(_source)
        _source["thread"].start()

def unit_test_yolov5s(loadso='libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so', sensor='2'):

    load([
        loadso,
        '-p', '/home/config/yolov5s.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()
    load([
        loadso,
        '-p', '/home/config/yolov5s_face.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()

def unit_test_ax_pose(loadso='libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so', sensor='2'):

    load([
        loadso,
        '-p', '/home/config/ax_pose.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()
    load([
        loadso,
        '-p', '/home/config/hrnet_pose.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()

def unit_test_hand_pose(loadso='libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so', sensor='2'):

    load([
        loadso,
        '-p', '/home/config/hand_pose.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()
    load([
        loadso,
        '-p', '/home/config/hand_pose_yolov7_palm.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()

def unit_test_display(loadso='libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so', sensor='2'):

    load([
        loadso,
        # '-p', '/home/config/pp_human_seg.json',
        # '-p', '/home/config/yolo_fastbody.json',
        # '-p', '/home/config/hrnet_animal_pose.json',
        # '-p', '/home/config/yolopv2.json',
        # '-p', '/home/config/license_plate_recognition.json',
        # '-p', '/home/config/yolov5s_face_recognition.json',
        '-p', '/home/config/yolov5s.json',
        '-c', sensor,
    ])

    lcd_width, lcd_height = 854, 480
    from PIL import Image, ImageDraw
    logo = Image.open("/home/res/logo.png")
    img = Image.new('RGBA', (lcd_width, lcd_height))
    ui = ImageDraw.ImageDraw(img)
    ui.rectangle((54,40,800,400), fill=(0,0,0,0), outline=(255,0,0,100), width=100)
    img.paste(logo, box=(lcd_width//2, lcd_height//2), mask=None)
    config("display", (lcd_width, lcd_height, "rgba", img.tobytes()))
    # config("input", False)
    # for i in range(300):
    #     time.sleep(0.01)
    #     tmp = result()
    #     if tmp:
    #         print(tmp)
    # config("input", True)
    # for i in range(100):
    #     time.sleep(0.001)
    #     tmp = result()
    #     if tmp:
    #         print(tmp)
    #     ai = config("camera")
    #     if ai and ai.mode == "RGB":
    #         rgba = img.copy()
    #         tmp = Image.frombytes("RGB", (ai.width, ai.height), ai.data)
    #         tmp.thumbnail((ai.width // 2, ai.height // 2))
    #         rgba.paste(tmp, box=(0, 0), mask=None)
    #         config("display", (lcd_width, lcd_height, "rgba", rgba.tobytes()))
    config("input", False)
    config("hide", True)
    ui = ImageDraw.ImageDraw(img)
    ui.rectangle((0,0,lcd_width, lcd_height), fill=(0,0,0,0), outline=(0,255,0,100), width=100)
    img.paste(logo, box=(lcd_width//2, lcd_height//2), mask=None)
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            rgba = img.copy()
            ui = ImageDraw.ImageDraw(rgba)
            for i in tmp['mObjects']:
                x = i['bbox']['x'] * lcd_width
                y = i['bbox']['y'] * lcd_height
                w = i['bbox']['w'] * lcd_width
                h = i['bbox']['h'] * lcd_height
                objname = i['objname']
                objprob = i['prob']
                ui.rectangle((x,y,x+w,y+h), fill=(255,0,0,100), outline=(0,0,255,100))
                ui.text((x,y), objname, fill=(255,0,0,100))
                ui.text((x,y+20), str(objprob), fill=(255,0,0,100))
            config("display", (lcd_width, lcd_height, "rgba", rgba.tobytes()))
    config("hide", False)
    config("output", False)
    drop()

def unit_test_yolov5s_seg(loadso='libsample_vin_ivps_joint_venc_rtsp_vo_sipy.so', sensor='2'):

    load([
        loadso,
        # '-p', '/home/config/pp_human_seg.json',
        # '-p', '/home/config/yolo_fastbody.json',
        # '-p', '/home/config/hrnet_animal_pose.json',
        # '-p', '/home/config/yolopv2.json',
        # '-p', '/home/config/license_plate_recognition.json',
        # '-p', '/home/config/yolov5s_face_recognition.json',
        '-p', '/home/config/yolov5_seg.json',
        '-c', sensor,
    ])
    for i in range(300):
        time.sleep(0.01)
        tmp = result()
        if tmp and tmp['nObjSize']:
            for i in tmp['mObjects']:
                print(i)
    drop()

def unit_test():
    unit_test_display()
    unit_test_ax_pose()
    unit_test_yolov5s(loadso='libsample_vin_ivps_joint_vo_sipy.so')
    unit_test_hand_pose()
    unit_test_yolov5s_seg(loadso='libsample_vin_ivps_joint_vo_sipy.so')

if __name__ == "__main__":
    unit_test()
