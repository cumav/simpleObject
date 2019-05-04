import time

from SimpleObjectDetection import SimpleObjectDetection

x = SimpleObjectDetection()
x.image_boxes("test.png", plot=True)

time.sleep(100)