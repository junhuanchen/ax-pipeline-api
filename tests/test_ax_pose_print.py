
import time
from ax import pipeline
from PIL import Image, ImageDraw
lcd_width, lcd_height = 854, 480
logo = Image.open("/home/res/logo.png")
img = Image.new('RGBA', (lcd_width, lcd_height), (255,0,0,200))
ui = ImageDraw.ImageDraw(img)
ui.rectangle((20,20,lcd_width-20,lcd_height-20), fill=(0,0,0,0), outline=(0,0,255,100), width=20)
img.paste(logo, box=(lcd_width-logo.size[0], lcd_height-logo.size[1]), mask=None)
r,g,b,a = img.split()
canvas_argb = Image.merge("RGBA", (a,b,g,r))
# ready sipeed logo canvas
import threading
def print_data(threadName, delay):
    print("print_data 1", threadName, pipeline.work())
    while pipeline.work() is False: # wait work
        time.sleep(delay)
    # pipeline.config("hide", True)
    while pipeline.work():
        # time.sleep(delay)
        argb = canvas_argb.copy()
        tmp = pipeline.result()
        if tmp and tmp['nObjSize']:
            ui = ImageDraw.ImageDraw(argb)
            for i in tmp['mObjects']:
                if i["bHasBoxVertices"]:
                    points = [ (p['x'] * lcd_width, p['y'] * lcd_height) for p in i['bbox_vertices']]
                    ui.polygon(points, fill=(100,0,0,255), outline=(255,0,0,255))
                else:
                    x = i['bbox']['x'] * lcd_width
                    y = i['bbox']['y'] * lcd_height
                    w = i['bbox']['w'] * lcd_width
                    h = i['bbox']['h'] * lcd_height
                    ui.rectangle((x,y,x+w,y+h), fill=(100,0,0,255), outline=(255,0,0,255))
                if i["bHasLandmark"]:
                    for p in i["landmark"]:
                        x, y = (int(p['x']*lcd_width), int(p['y']*lcd_height))
                        ui.rectangle((x-2,y-2,x+2, y+2), outline=(255,0,0,255))
        pipeline.config("ui_image", (lcd_width, lcd_height, "ABGR", argb.tobytes()))
    print("print_data 2", pipeline.work())

thread = threading.Thread(target=print_data, args=("Thread-1", 0.01, ))
thread.start()

pipeline.load([
    b'libsample_vin_ivps_joint_vo_sipy.so',
    b'-p', b'/home/config/ax_pose.json',
    # b'-p', b'/home/config/hand_pose.json',
    # b'-p', b'/home/config/yolov5s_face.json',
    b'-c', b'0',
])

thread.join()