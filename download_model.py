from ultralytics import YOLO
#import shutil

model = YOLO('yolov8m.pt')
'''shutil.copy(model.ckpt_path, 'models/yolov8n.pt')
print("✅ โหลดและย้ายไฟล์เสร็จเรียบร้อยแล้ว -> models/yolov8n.pt")'''
print(model.names)