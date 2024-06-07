
import argparse
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from ultralytics import YOLOv10
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

torch.cuda.is_available()
track_history = defaultdict(list)

current_region = None

class Yolov8_Video():
    
    def __init__(self, onnx_model="Models/yolov8n.onnx", input_image="Torres.jpg", confidence_thres=0.5, iou_thres=0.5):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
    
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        #self.classes = yaml_load(check_yaml("coco8.yaml"))["names"]
        #self.classes = names
        # Generate a color palette for the classes
        #self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
    def run(
    model="Models/yolov10n.pt",
    source="Londres.mp4",
    device="0",
    view_img=True,
    save_img=True,
    exist_ok=False,
    classes=None,
    line_thickness=2,
    track_thickness=2,
    region_thickness=2,
):
        """
        Run Region counting on a video using YOLOv8 and ByteTrack.

        Supports movable region for real time counting inside specific area.
        Supports multiple regions counting.
        Regions can be Polygons or rectangle in shape

        Args:
            weights (str): Model weights path.
            source (str): Video file path.
            device (str): processing device cpu, 0, 1
            view_img (bool): Show results.
            save_img (bool): Save results.
            exist_ok (bool): Overwrite existing files.
            classes (list): classes to detect and track
            line_thickness (int): Bounding box thickness.
            track_thickness (int): Tracking line thickness
            region_thickness (int): Region thickness.
        """
        vid_frame_count = 0

        # Check source path
        if not Path(source).exists():
            raise FileNotFoundError(f"Source path '{source}' does not exist.")

        # Setup Model
        model = YOLOv10("yolov8n.pt")
        model.to("cuda") if device == "0" else model.to("cpu")
        #model.fuse()
        

        # Extract classes names
        names = model.model.names
    
        print(names)
        # Video setup
        videocapture = cv2.VideoCapture(source)
        frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
        fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

        # Output setup
        save_dir = "video_outputs"
        
        video_writer = cv2.VideoWriter(save_dir + f"{Path(source).stem}.mp4", fourcc, fps, (frame_width, frame_height))

        # Iterate over video frames
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                break
            vid_frame_count += 1

            # Extract the results
            results = model.track(frame, persist=True, classes=list(names),conf=0.9, iou=0.9)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()

                annotator = Annotator(frame, line_width=line_thickness, example=str(names))

                for box, track_id, cls in zip(boxes, track_ids, clss):
                    annotator.box_label(box, f'{str(names[cls])}:{track_id}', color=colors(cls, True))
                    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                    track = track_history[track_id]  # Tracking Lines plot
                    track.append((float(bbox_center[0]), float(bbox_center[1])))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)

                    # Check if detection inside region
                    #for region in counting_regions:
                     #   if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                      #      region["counts"] += 1
            
            if view_img:
                if vid_frame_count == 1:
                    cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
                    #cv2.setMouseCallback("Ultralytics YOLOv8 Region Counter Movable", mouse_callback)
                cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

            if save_img:
                video_writer.write(frame)

           # for region in counting_regions:  # Reinitialize count for each region
            #    region["counts"] = 0

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        """
         # Draw regions (Polygons/Rectangles)
            for region in counting_regions:
                region_label = str(region["counts"])
                region_color = region["region_color"]
                region_text_color = region["text_color"]

                polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
                centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

                text_size, _ = cv2.getTextSize(
                    region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
                )
                text_x = centroid_x - text_size[0] // 2
                text_y = centroid_y + text_size[1] // 2
                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    region_color,
                    -1,
                )
                cv2.putText(
                    frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
                )
                cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)
        """


        del vid_frame_count
        video_writer.release()
        videocapture.release()
        cv2.destroyAllWindows()
        
objet = Yolov8_Video().run()