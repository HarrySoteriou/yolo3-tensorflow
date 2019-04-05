import os
import config
import argparse
import numpy as np
import tensorflow as tf
from yolo_predict import yolo_predictor
from PIL import Image, ImageFont, ImageDraw
from utils import letterbox_image, load_weights
from keras import backend as K


##IMPORT TIME TO MEASURE RUNNING TIME
import time

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index

### we create a list with the titles of all the pictures of our dataset
name_space = os.listdir(config.dataset)
#print(name_space)


# print(name_space)

##USE MULTITHREADING AND GPUS https://medium.com/@lisulimowicz/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
### I HAVE COMMENTED OUT EVERYTHING THAT HAD TO TO WITH OUTPUTING AN IMAGE WITH THE DETECTION SQUARES
### OUTPUTS NUMBER OF OBJECTS, CLASS AND ACCURACY SCORES, ALONG WITH RUNNING TIME

def detect(image_path, model_path, yolo_weights=None):
    start_time = time.time()
    for image_id in name_space[0:5]:
        reset_time = time.time()
        tf.reset_default_graph()

        reset_end = time.time()
        st_image_prep = time.time()

        image = Image.open(image_path + image_id) #variable was: image

        resize_image = letterbox_image(image, (416, 416)) # should be (320, 240)
        image_data = np.array(resize_image, dtype=np.float32)
        image_data /= 255.
        image_data = np.expand_dims(image_data, axis=0)
        input_image_shape = tf.placeholder(dtype=tf.int32, shape=(2,))
        input_image = tf.placeholder(shape=[None, 416, 416, 3], dtype=tf.float32)
        predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.classes_path, config.anchors_path)
        # boxes, scores, classes = predictor.predict(input_image, input_image_shape)

        en_image_prep = time.time()

        with tf.Session() as sess:
            st1_sess = time.time()
            if yolo_weights is not None:
                with tf.variable_scope('predict'):
                    st1_end = time.time()
                    st2_sess = time.time()
                    boxes, scores, classes = predictor.predict(input_image, input_image_shape)
                    st2_end = time.time()
                    st3_sess = time.time()
                    load_op = load_weights(tf.global_variables(scope='predict'), weights_file=yolo_weights)
                    st3_end = time.time()
                    st4_sess = time.time()
                    sess.run(load_op)
                    st4_end = time.time()
                    end = time.time()

                    print('Image preprocessing takes: {}'.format(en_image_prep - st_image_prep))
                    print('tf.reset_default_graph takes: {}'.format(reset_end - reset_time))
                    print('if not None statement an with... time is: {}'.format(st1_end -st1_sess))
                    print('boxes, scores, classes time is: {}'.format(st2_end -st2_sess))
                    print('load_op = load_weights(tf.global time is: {}'.format(st3_end -st3_sess))
                    print('sess.run(load_op) time is: {}'.format(st4_end -st4_sess))

            else:
                print('session not running')
                saver = tf.train.Saver()
                saver.restore(sess, model_path)

            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })

            print('Found {} boxes for {}'.format(len(out_boxes), 'image: ', image_id))
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                      size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i, c in reversed(list(enumerate(out_classes))):
                predicted_class = predictor.class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                label = '{} {:.2f}'.format(predicted_class, score)
                print(label)

                print("--- %s seconds ---" % (time.time() - start_time))

                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                print(label, (left, top), (right, bottom))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=predictor.colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=predictor.colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            image.show()
            image.save('./result1.jpg')


"""
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            #TODO make the draw function below loop over every box found until none are left
            ## My kingdom for a good redistributable image drawing library.
           for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=predictor.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw """




if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default = argparse.SUPPRESS)
    parser.add_argument(
        '--image_file', type = str, help = 'image file path'
    )
    FLAGS = parser.parse_args()


    if config.pre_train_yolo3 == True:
        detect(config.dataset, config.model_dir, config.yolo3_weights_path)
    else:
        detect(config.dataset, config.model_dir)
