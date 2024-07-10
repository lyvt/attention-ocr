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
