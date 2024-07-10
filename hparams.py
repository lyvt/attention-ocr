import os

class Hparams:
	def __init__(self, data_dir, base_model_name, save_path, rnn_cell, rnn_unit, batch_size, number_epochs, learning_rate,  test_ckpt, infer_ckpt):
		self.train_record_path = os.path.join (data_dir, 'data.train')
		self.valid_record_path = os.path.join (data_dir, 'data.val')
		self.test_record_path = os.path.join (data_dir, 'data.test')
		
		self.charset_path = 'preprocess/charsets/charset_size=94.txt'
		self.save_path = save_path
		self.save_best = False
		self.max_to_keep = 1000

		### input params
		self.image_shape = (40, 416, 3) 
		self.nul_code = 93
		self.charset_size = 94
		self.max_char_length = 10

		### conv_tower params

		self.base_model_name = base_model_name #InceptionV3 , InceptionResNetV2, MobileNet
		self.use_encode_cordinate = True


		### RNN tower params
		self.rnn_cell = rnn_cell #bilstm, gru, lstm
		self.rnn_units = rnn_unit
		self.dense_units = 256
		self.weight_decay = 0.00004

		### attention params
		self.model_size = 512
		self.num_heads = 8
		self.attention = "Baduhade"

		### training params
		self.batch_size = batch_size
		self.max_epochs = number_epochs
		self.lr =  learning_rate
		self.test_ckpt = test_ckpt
		self.infer_ckpt = infer_ckpt

"""
params:
For Training:
	data_dir: thu muc luu tru du lieu da qua xu ly
	base_model_name: loaij backbone, support 3 loai sau: InceptionV3 , InceptionResNetV2, MobileNet
	save_path: thu muc luu tru checkpoint
	rnn_cell: loai rnn cell, support 3 loai sau: bilstm, gru, lstm
	rnn_unit: (int) so luong no ron cua lop rnn
	batch_size: (int) kich thuoc batch de train
	number_epochs: (int) so luong epoch
	learning_rate: (float) toc do hoc
For test and inference:
	test_ckpt: (int) so thu tu cua checkpoint được dùng để test, cần thêm các augment của training như: data_dir, base_model_name, save_path, rnn_cell, rnn_unit
	infer_ckpt: (int) so thu tu cua checkpoint được dùng để inference , cần thêm các augment của training như: base_model_name, save_path, rnn_cell, rnn_unit
"""

data_dir = 'processed_data/ds01'
base_model_name = 'MobileNet' 
save_path = 'runs/MobileNet'
rnn_cell='lstm' 
rnn_unit = 256
batch_size = 32
number_epochs = 10 
learning_rate = 0.001  
test_ckpt = 1
infer_ckpt = 1

hparams = Hparams(data_dir, base_model_name, save_path, rnn_cell, rnn_unit, batch_size, number_epochs, learning_rate,  test_ckpt, infer_ckpt)

#Cấu hình mô hình mẫu
"""
data_dir = 'processed_data/ds01'
base_model_name = 'InceptionV3' 
save_path = 'runs/20240708_InceptionV3'
rnn_cell='lstm' 
rnn_unit = 256
infer_ckpt = 100
"""

