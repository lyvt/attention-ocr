from preprocess.tools import *

raw_dir = '/data1/home/lyvt/ocr/data/scanit_data/full_ver8/train' #label_train_all_correct.txt
label_dir = '/data1/home/lyvt/ocr/data/scanit_data/full_ver8/label_train_all_correct.txt'


#convert dataset to serial for quickly training:
charset_path = 'preprocess/charsets/charset_size=94.txt' # thư mục lưu charset 
save_dir = 'processed_data/ds01' # thư mục lưu dữ liệu sau xử lý

num_train, num_val, num_test = gen_record (raw_dir, label_dir, charset_path, save_dir, 0.8)