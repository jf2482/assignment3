import jetson_inference
import jetson_utils
import cv2
import numpy as np
import os
image_paths = [
    "/home/nvidia/jetson-inference/data/images/p0.jpg",
    "/home/nvidia/jetson-inference/data/images/p1.jpg"
]
thresholds = [0.1, 0.9]
MODEL_PATH = "/home/nvidia/jetson-inference/data/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff"
LABELS_PATH = "/home/nvidia/jetson-inference/data/networks/ssd_coco_labels.txt"
for img_path in image_paths:
    if not os.path.exists(img_path):
        print(f"can not find: {img_path}")
        continue

    for th in thresholds:
        print(f"\n============================================================")
        print(f" Testing Image: {img_path}")
        print(f" Threshold: {th}")
        print(f"============================================================")
            
        net = jetson_inference.detectNet(
            model=MODEL_PATH, 
            labels=LABELS_PATH, 
            threshold=th,
            input_blob="Input",
            output_cvg="NMS",
            output_bbox="NMS_1"
        )
           
        camera = jetson_utils.videoSource(img_path)
        img = camera.Capture()
        
        if img is None:
            print(f"please check the path: {img_path}")
            continue
              
        detections = net.Detect(img)
        
        print(f"Number of detected objects: {len(detections)}")
        for i, detection in enumerate(detections):
            print(f"--- Detection {i+1} ---")
            print(f"-- ClassID: {detection.ClassID}")
            print(f"-- Confidence: {detection.Confidence:.6f}")
            print(f"-- Left: {detection.Left:.5f}")
            print(f"-- Top: {detection.Top:.5f}")
            print(f"-- Right: {detection.Right:.5f}")
            print(f"-- Bottom: {detection.Bottom:.5f}")
            print(f"-- Width: {detection.Width:.5f}")
            print(f"-- Height: {detection.Height:.5f}")
            print(f"-- Area: {detection.Area:.1f}")
            print(f"-- Center: ({detection.Center[0]:.3f}, {detection.Center[1]:.3f})")
             
        #img_name = img_path.split('/')[-1].split('.')[0]
        #output_name = f"result_{img_name}_th{th}.jpg"
        
        #display = jetson_utils.videoOutput(output_name)
        #display.Render(img)
        #display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
        
        #print(f"saved as: {output_name}")
        #print(f"============================================================\n")
 
        img_np = jetson_utils.cudaToNumpy(img)
        
        if img_np.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        for detection in detections:
            left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
                 
            cv2.rectangle(img_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            
            class_name = net.GetClassDesc(detection.ClassID)
            label = f"{class_name} {detection.Confidence:.2f}"
              
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img_bgr, (left, top - text_height - 5), (left + text_width, top), (0, 255, 0), -1)  
            
            cv2.putText(img_bgr, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        img_name = os.path.basename(img_path).split('.')[0]
        output_name = f"result_{img_name}_th{th}.jpg"
        
        cv2.imwrite(output_name, img_bgr)
        print(f"good: {output_name}")
