def filter_detections(results, target_classes, conf_threshold=0.6):
    """กรองเฉพาะคลาสที่ต้องการ และค่าความมั่นใจสูงกว่า threshold"""
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls in target_classes and conf >= conf_threshold:
            detections.append(box.xyxy[0].tolist())
    return detections