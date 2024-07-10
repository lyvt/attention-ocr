import tensorflow as tf
import numpy as np
import cv2
import glob
import os

from model import Model
from hparams import hparams
os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

def resize (im):
    max_w =  416
    new_h = 40
    h, w, d = im.shape
    unpad_im = cv2.resize(im, (int(new_h*w/h), new_h), interpolation = cv2.INTER_AREA)
    if unpad_im.shape[1] > max_w:
        pad_im = cv2.resize(im, (max_w, new_h), interpolation = cv2.INTER_AREA)
    else:
        pad_im = cv2.copyMakeBorder(unpad_im,0,0,0,max_w-int(new_h*w/h),cv2.BORDER_CONSTANT,value=[0,0,0])
    return pad_im

def predict_one_image (im, model):
    image = resize (im)
    result = model.inference (image/255.0)
    result = result.rstrip ('<nul>')
    return result


def predict_label(image_folder):
    model = Model()
    model.create_model()
    model.load_weight(hparams.infer_ckpt)
    
    num_name = 0

    for f in glob.glob(os.path.join (image_folder, "*.*")):
        try:
            im = cv2.imread (f)
            if im is None:
                continue
            ocr_result = predict_one_image (im, model)
            print (ocr_result)
         
        except Exception as e: print(e)

image_folder = '/data1/home/lyvt/ocr/temp/parseq/demo_images'
predict_label (image_folder)