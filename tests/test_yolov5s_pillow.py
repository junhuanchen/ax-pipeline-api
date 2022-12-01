
import time
import pipeline
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
                x = i['bbox']['x'] * lcd_width
                y = i['bbox']['y'] * lcd_height
                w = i['bbox']['w'] * lcd_width
                h = i['bbox']['h'] * lcd_height
                objlabel = i['label']
                objprob = i['prob']
                ui.rectangle((x,y,x+w,y+h), fill=(100,0,0,255), outline=(255,0,0,255))
                ui.text((x,y), str(objlabel))
                ui.text((x,y+20), str(objprob))
        pipeline.config("ui_image", (lcd_width, lcd_height, "ABGR", argb.tobytes()))
    print("print_data 2", pipeline.work())

thread = threading.Thread(target=print_data, args=("Thread-1", 0.01, ))
thread.start()

pipeline.load([
    b'libsample_vin_ivps_joint_vo_sipy.so',
    b'-m', b'/home/models/yolov5s.joint',
    b'-p', b'/home/config/yolov5s.json',
    b'-c', b'0',
])

thread.join()