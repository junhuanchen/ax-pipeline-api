
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

def rgba2argb(rgba):
    r,g,b,a = rgba.split()
    return Image.merge("RGBA", (a,b,g,r))
canvas_argb = rgba2argb(img)

pipeline.load([
    'libsample_vin_ivps_joint_vo_sipy.so',
    '-p', '/home/config/ax_pose.json',
    # '-p', '/home/config/hand_pose.json',
    # '-p', '/home/config/yolov5s_face.json',
    '-c', '2',
])

while pipeline.work():
    time.sleep(0.001)
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
            for p in i["landmark"]:
                x, y = (int(p['x']*lcd_width), int(p['y']*lcd_height))
                ui.rectangle((x-4,y-4,x+4, y+4), outline=(255,0,0,255))
    pipeline.config("ui_image", (lcd_width, lcd_height, "ARGB", argb.tobytes()))

pipeline.drop()
