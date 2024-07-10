import tensorflow as tf
import argparse
import cv2
import os
import random
import pdb
import numpy as np

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

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(image_encoded, image_format, image_class, image_text):
    feature = {
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
        'image/class': _int64_list_feature(image_class),
        'image/text': _bytes_feature(image_text)
    }
  
    # Create a Features message using tf.train.Example.
  
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def gen_record (raw_dir, label_path, charset_path, save_dir, ratio, use_augment = True):
    # read image filenames
    os.makedirs (save_dir, exist_ok = True)
    filenames = []
    texts = []
    with open(label_path, "r") as f:
        for row in f:
            if "\t" not in row:
                continue
            split_row = row[:-1].split("\t")
            filenames.append(split_row[0])
            texts.append(split_row[1])

    #shuffle:
    if ratio <=0 or ratio >= 1:
        return None
    temp = list(zip(filenames, texts))
    random.shuffle(temp)
    filenames, texts = zip(*temp)
    filenames, texts = list(filenames), list(texts)
    idx_test = int (ratio * len (filenames))
    idx_val = idx_test + int (0.1*(1- ratio) * len (filenames))
    test_filenames, test_texts = filenames [:idx_test], texts [: idx_test]
    val_filenames, val_texts = filenames [idx_test:idx_val], texts [idx_test: idx_val]
    train_filenames, train_texts = filenames [idx_val:], texts [idx_val: ]


    charset = {}
    with open(charset_path, "r") as f:
        for row in f:
            val, key = row[:-1].split("\t")
            charset[key] = int(val)

    num_train = convert_data (raw_dir, train_filenames, train_texts, charset, os.path.join (save_dir, 'data.train'))
    num_val = convert_data (raw_dir, val_filenames, val_texts, charset, os.path.join (save_dir, 'data.val'))
    num_test = convert_data (raw_dir, test_filenames, test_texts, charset, os.path.join (save_dir, 'data.test'))
    return num_train, num_val, num_test

def convert_data (raw_dir, filenames, texts, charset, save_path):
    null_id = 93
    max_seqlen = 10
    
    cnt = 0
    error_files = []
    with tf.io.TFRecordWriter(save_path) as writer:
        for i, (filename, text) in enumerate(zip(filenames, texts )):
            if len(text) > max_seqlen:
                    continue
            ### prepare all feature values
            # image/encoded
            try:
                im = cv2.imread(os.path.join(raw_dir, filename))
                #example
                # im_encode_1 = cv2.imencode('.png', im)[1] 
                # data_encode = np.array(im_encode_1) 
                # im_encode_1 = data_encode.tobytes() 
                # with open(os.path.join(raw_dir, filename), "rb") as f:
                #     im_encode_2 = f.read()
                #     pdb.set_trace()
                
                if im is None:
                    continue
                pad_im = resize (im)
                img_encode = cv2.imencode('.png', pad_im)[1] 
                
                data_encode = np.array(img_encode) 
                image_encoded = data_encode.tobytes() 
                # image/format
                image_format = "png".encode()
            
                cnt += 1
                image_class = []
                for char in text:
                    image_class.append(charset[char])
                while len(image_class) < max_seqlen:
                    image_class.append(null_id)
            
                image_text = text.encode()
                # write to TFRecordFile
                example = serialize_example(image_encoded,image_format, image_class, image_text)
                writer.write(example)
                print ("done")
            except Exception as e: print(e)

    print(cnt)
    return cnt

