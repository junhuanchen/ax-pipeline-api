
def replay_test():
    import m3axpi
    import time
    for i in range(100):
        time.sleep(0.05)
        print(len(m3axpi.forward()))
    m3axpi.camera(640, 360, 2)
    m3axpi.load("/home/config/xxxxx.json")
    for i in range(100):
        time.sleep(0.05)
        print(len(m3axpi.forward()))
    m3axpi.camera(640, 360, 0)
    m3axpi.load("/home/config/yolov7.json")
    for i in range(100):
        time.sleep(0.05)
        print(len(m3axpi.forward()))
    m3axpi.camera(640, 360, 2)
    m3axpi.load("/home/config/hand_pose.json")
    for i in range(100):
        time.sleep(0.05)
        print(len(m3axpi.forward()))
# replay_test()

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
    
    res = m3axpi.forward()
    if 'nObjSize' in res:
        ui = ImageDraw.ImageDraw(rgba)
        for obj in res['mObjects']:
            x, y, w, h = int(obj['bbox'][0]*lcd_width), int(obj['bbox'][1]*lcd_height), int(obj['bbox'][2]*lcd_width), int(obj['bbox'][3]*lcd_height)
            ui.rectangle((x,y,x+w,y+h), fill=(255,0,0,100), outline=(255,0,0,255))
            ui.text((x, y), "%s:%02d" % (obj['objname'], obj['prob']*100), font=fnt)
            rgba.paste(logo, box=(x+w-logo.size[1], y+h-logo.size[1]), mask=None)

            rgb_ui = ImageDraw.ImageDraw(rgb)
            x, y, w, h = int(obj['bbox'][0]*tmp[1]), int(obj['bbox'][1]*tmp[0]), int(obj['bbox'][2]*tmp[1]), int(obj['bbox'][3]*tmp[0])
            rgb_ui.rectangle((x,y,x+w,y+h), outline=(255,0,0,255))
            rgb_ui.text((x, y), "%s:%02d" % (obj['objname'], obj['prob']*100), font=fnt)

    rgba.paste(rgb, box=(0, 0), mask=None)

    m3axpi.display([lcd_height, lcd_width, lcd_channel, rgba.tobytes()])
