from functools import partial

import cv2
import gradio as gr
from PIL import Image
from ultralytics import YOLO
from Yolov8 import YOLOv8
from Yolov8_Interface import Yolov8_Interface

baseModel= YOLO("./Models/yolov8n.onnx")

def run_image(image):
    image.save("uploaded_image.jpg")
    results = baseModel.predict(image,name="Test_result",project="/var/proyectos/YoloWorldTest/volumeTest/examples", save=True, imgsz=120, conf=0.5,classes=[0])
    boxes = results[0].boxes.xywh.cpu()
    for box in boxes:
        x, y, w, h = box
    print(f"Width of Box: {w}, Height of Box: {h}")
    print("****************************************")
    print(results)
    return Image.open(f"Test_result/{results[0].path}")



def demo(string,olo_interface:Yolov8_Interface):
    with gr.Blocks(title="YOLO-World") as demo:
        with gr.Row():
            gr.Markdown('<h1><center>YOLO-World: Real-Time Open-Vocabulary '
                        'Object Detector</center></h1>')
        
        with gr.Row():
            with gr.Column(scale=0.3):
                with gr.Row():
                    image = gr.Image( label='input image')
                    
                input_text = gr.Textbox(
                    lines=7,
                    label='Enter the classes to be detected, ''separated by comma',
                    value=" ".join(list(baseModel.names.values())),
                    elem_id='textbox')
                
                model = gr.Dropdown(label="Choose an Option", choices=["yolov8n.onnx", "yolov8n.pt", "yolov8n-seg.pt"])
                
                with gr.Row():
                    submit = gr.Button('Submit')
                    clear = gr.Button('Clear')
                    confidence_thres = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.05,
                        step=0.001,
                        interactive=True,
                        label='Confidence Threshold')
                    nms_thr = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.001,
                        interactive=True,
                        label='NMS Threshold')
            with gr.Column(scale=0.8):
                output_image = gr.Image(
                    label='output image')
            
                
        submit.click(fn=partial(YOLOvTest.main),inputs=[model,image,confidence_thres, nms_thr],outputs=[output_image])
        clear.click(lambda: [[], '', ''], None,
                    [image, input_text, output_image])
        
        demo.launch(server_name='0.0.0.0')    
YOLOvTest = YOLOv8()
demo("lechuga",YOLOvTest)
