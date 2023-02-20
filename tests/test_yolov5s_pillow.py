
import time
from ax import pipeline
from PIL import Image, ImageDraw

# ready sipeed logo canvas
lcd_width, lcd_height = 854, 480

img = Image.new('RGBA', (lcd_width, lcd_height), (255,0,0,200))
ui = ImageDraw.ImageDraw(img)
ui.rectangle((20,20,lcd_width-20,lcd_height-20), fill=(0,0,0,0), outline=(0,0,255,100), width=20)

logo = Image.open("/home/res/logo.png")
img.paste(logo, box=(lcd_width-logo.size[0], lcd_height-logo.size[1]), mask=None)

pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/config/yolov5s.json',
    # '-p', '/home/config/yolov8.json',
    '-c', '2',
])

while pipeline.work():
    time.sleep(0.001)
    rgba = img.copy()
    tmp = pipeline.result()
    if tmp and tmp['nObjSize']:
        ui = ImageDraw.ImageDraw(rgba)
        for i in tmp['mObjects']:
            x = i['bbox']['x'] * lcd_width
            y = i['bbox']['y'] * lcd_height
            w = i['bbox']['w'] * lcd_width
            h = i['bbox']['h'] * lcd_height
            objlabel = i['label']
            objprob = i['prob']
            ui.rectangle((x,y,x+w,y+h), fill=(255,0,0,100), outline=(255,0,0,100))
            ui.text((x,y), str(objlabel))
            ui.text((x,y+20), str(objprob))
    pipeline.config("display", (lcd_width, lcd_height, "rgba", rgba.tobytes()))

pipeline.drop()
