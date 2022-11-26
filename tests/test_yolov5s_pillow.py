
try:
    import time
    import pipeline
    lcd_width, lcd_height = 854, 480
    from PIL import Image, ImageDraw
    logo = Image.open("/home/res/logo.png")
    img = Image.new('RGBA', (lcd_width, lcd_height), (255,0,0,200))
    ui = ImageDraw.ImageDraw(img)
    ui.rectangle((20,20,lcd_width-20,lcd_height-20), fill=(0,0,0,0), outline=(0,0,255,100), width=20)
    img.paste(logo, box=(lcd_width-logo.size[0], lcd_height-logo.size[1]), mask=None)
    r,g,b,a = img.split()
    src_argb = Image.merge("RGBA", (a,b,g,r))
    
    import threading
    def print_data(threadName, delay):
        print("print_data 1", threadName, pipeline.work())
        while pipeline.work() is False: # wait work
            time.sleep(delay)
        # pipeline.config("ai_hide", False)
        while pipeline.work():
            # time.sleep(delay)
            tmp = pipeline.result()
            if tmp and tmp['nObjSize']:
                src_argb = Image.merge("RGBA", (a,b,g,r))
                ui = ImageDraw.ImageDraw(src_argb)
                for i in tmp['mObjects']:
                    x = i['bbox']['x'] * lcd_width
                    y = i['bbox']['y'] * lcd_height
                    w = i['bbox']['w'] * lcd_width
                    h = i['bbox']['h'] * lcd_height
                    objlabel = i['label']
                    objprob = i['prob']
                    ui.rectangle((x,y,x+w,y+h), fill=(100,0,0,255), outline=(255,0,0,255)) # fill fps:20>16
                    ui.text((x,y), str(objlabel))
                    ui.text((x,y+20), str(objprob))
                pipeline.config("ui_image", pipeline._image(lcd_width, lcd_height, "ABGR", src_argb.tobytes()))
        # pipeline.config("ai_hide", True)
        print("print_data 2", pipeline.work())

    test = threading.Thread(target=print_data, args=("Thread-1", 0.01, ))
    test.start()

    pipeline.load([
        b'libsample_vin_ivps_joint_vo_sipy.so',
        b'-m', b'/home/models/yolov5s.joint',
        b'-p', b'/home/config/yolov5s.json',
        b'-c', b'0',
    ])

    test.join()
except Exception as e:
    import traceback
    traceback.print_exc()
    print(e, "maybe apt install python3-pil -y")