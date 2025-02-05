from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)  # for exponential backoff

from typing import Union
from fastapi import FastAPI, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import numpy as np

import matplotlib.pyplot as plt
import os
import time
import asyncio
from fastapi import APIRouter, Depends, status, Response
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tenacity import retry, stop_after_attempt, wait_exponential
from fastapi import FastAPI, UploadFile, File

from mrcnn.config import Config
from  mrcnn import utils
import  mrcnn.model as modellib
from  mrcnn import visualize
from  mrcnn.model import log
import pickle
import joblib
import cv2
from fastapi.responses import StreamingResponse
import io
import joblib
from mask import refine_masks ,resize_image,postProcess
from config import configuration


label_names=['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 

NUM_CATS = 46
IMAGE_SIZE = 512


class ImageUpload(BaseModel):
    file: UploadFile


@app.get("/health")
def get_request():
    """
    Dummy function to test if server is running.
    """
    return {"Hello": "MaskRCNN"}


@retry(wait= wait_exponential(min=60, max=120), stop=stop_after_attempt(6))
@app.post("/detect/")
async def display_image(file: UploadFile = File(...)):
    
    try:
        file_bytes = await file.read()
        np_array = np.frombuffer(file_bytes, np.uint8)

        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        model = configuration()
        result = model.detect([resize_image(img)])
        r = result[0]
        #print(r)

        masks,rois = postProcess(r,img)
        filtered_ids = [x for x in r['class_ids'] if x in {1, 2, 32, 34}]
        filtered_indices = [i for i, id in enumerate(r['class_ids']) if id in {1, 2, 32, 34}]

        # Apply the filter to masks, rois, and scores (if any)
        filtered_rois = rois[filtered_indices]
        filtered_masks = masks[:, :, filtered_indices]
        filtered_class_ids = np.array(filtered_ids)
        filtered_scores = r['scores'][filtered_indices]  # Make sure scores are also filtered
        assert filtered_rois.shape[0] == filtered_masks.shape[-1] == filtered_class_ids.shape[0], \
            f"Dimension mismatch: rois: {filtered_rois.shape[0]}, masks: {filtered_masks.shape[-1]}, class_ids: {filtered_class_ids.shape[0]}"

        
        fig, ax = plt.subplots(figsize=(12, 12))
        visualize.display_instances(
            img, filtered_rois, filtered_masks, filtered_class_ids,
            ['bg'] + label_names, filtered_scores,
            title='img1', ax=ax
        )
        
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)  
    
        return StreamingResponse(buf, media_type="image/png")

            
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    

    


