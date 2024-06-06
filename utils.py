import glob
import os
import re
import shutil
import time

from ultralytics import YOLO

dest_dir = "./Models"

def downloadModel(model:str):
    
    file_path = os.path.join(dest_dir,model)
    
    onxx_model = os.path.join(dest_dir,re.sub(r'\.\w*','.onnx',model))
    
    if os.path.isfile(onxx_model):
        return onxx_model
    
    if not os.path.isfile(file_path):   
        
        model_down = YOLO(model)
        model_down = model_down.export(format="onnx",)
        files_to_move = glob.glob(os.path.join("", model[:3] + '*'))
        
        for file in files_to_move:
            file_path = os.path.join(dest_dir,file)
            shutil.move(file, file_path)   
        onxx_model = os.path.join(dest_dir,re.sub(r'\.\w*','.onnx',model)) 
        return onxx_model
    
    

