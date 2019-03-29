num_parallel_calls = 4
input_shape = 416
max_boxes = 20
jitter = 0.3
hue = 0.1
sat = 1.0
cont = 0.8
bri = 0.1
norm_decay = 0.99
norm_epsilon = 1e-3
pre_train = True
num_anchors = 9
num_classes = 80
training = True
ignore_thresh = .5
learning_rate = 0.001
train_batch_size = 10
val_batch_size = 10
train_num = 118287
val_num = 5000
Epoch = 50
obj_threshold = 0.3
nms_threshold = 0.5

#IN ORDER TO UTILISE GPU NEED TO DOWNLOAD CUDA, https://julip.co/2009/09/how-to-install-and-configure-cuda-on-windows/
gpu_index = "1"

#THIS IS THE DIRECTORY WITH ALL OF THE SCRIPTS
path ='C:/Users/Harry Soteriou/Anaconda3/tensorflow-yolo3/'

log_dir = path+'logs/'
data_dir = path + 'model_data/'

##ADDED THIS ONE TO CREATE A LIST NAME WITH ALL OF THE FILE NAMES AS TO DETECT MULTIPLE FILES
dataset = path + 'dataset/'

#I DO NOT KNOW WHAT IS WRONG WITH THE MODEL_DIR
model_dir = path+'model/yolo3_model.py'
pre_train_yolo3 = True
yolo3_weights_path = path+'model_data/yolov3.weights'
darknet53_weights_path = path+'model_data/darknet53.weights'
anchors_path = path+'model_data/yolo_anchors.txt'
classes_path = path+'model_data/coco_classes.txt'

#THIS MATTER ONLY IF YOU TRY AND TRAIN THE MODEL
#IF WE WANT TO TRAIN THE MODEL OURSELVES
train_data_file = '/data0/dataset/coco/train2017'
val_data_file = '/data0/dataset/coco/val2017'
train_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_train2017.json'
val_annotations_file = '/data0/gaochen3/tensorflow-yolo3/annotations/instances_val2017.json'


