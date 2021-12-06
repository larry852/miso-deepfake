import os
import cv2
import math
import cv2
from mtcnn import MTCNN
import sys, os.path
import json
from keras import backend as K
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

base_path = './static/videos/'
checkpoint_filepath = '.././tmp_checkpoint'
model = 'vg.h5'

def _get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

def convert_video_to_image(filename):
    print(filename)
    if (filename.endswith(".mp4")):
        tmp_path = os.path.join(base_path, _get_filename_only(filename))
        print('Creating Directory: ' + tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        print('Converting Video to Images...')
        count = 0
        video_file = os.path.join(base_path, filename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5) #frame rate
        while(cap.isOpened()):
            frame_id = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frame_id % math.floor(frame_rate) == 0):
                print('Original Dimensions: ', frame.shape)
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, _get_filename_only(filename)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
        print("Done!")

def crop_faces(filename):
    tmp_path = os.path.join(base_path, _get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)
    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    faces_path = os.path.join(tmp_path, 'faces')
    print('Creating Directory: ' + faces_path)
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')

    for frame in frame_images:
        print('Processing ', frame)
        detector = MTCNN()
        image = cv2.cvtColor(cv2.imread(os.path.join(tmp_path, frame)), cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        print('Face Detected: ', len(results))
        count = 0
        
        for result in results:
            bounding_box = result['box']
            print(bounding_box)
            confidence = result['confidence']
            print(confidence)
            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3  # 30% as the margin
                margin_y = bounding_box[3] * 0.3  # 30% as the margin
                x1 = int(bounding_box[0] - margin_x)
                if x1 < 0:
                    x1 = 0
                x2 = int(bounding_box[0] + bounding_box[2] + margin_x)
                if x2 > image.shape[1]:
                    x2 = image.shape[1]
                y1 = int(bounding_box[1] - margin_y)
                if y1 < 0:
                    y1 = 0
                y2 = int(bounding_box[1] + bounding_box[3] + margin_y)
                if y2 > image.shape[0]:
                    y2 = image.shape[0]
                print(x1, y1, x2, y2)
                crop_image = image[y1:y2, x1:x2]
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, _get_filename_only(frame)), count)
                count = count + 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
            else:
                print('Skipped a face..')

def predict(filename):    
    input_size = 128
    batch_size_num = 32
    data_path = os.path.join(os.path.join(base_path, _get_filename_only(filename)))
    print(data_path)

    datagen = ImageDataGenerator(
        rescale = 1/255    #rescale the tensor values to [0,1]
    )

    generator = datagen.flow_from_directory(
        directory = data_path,
        target_size = (input_size, input_size),
        color_mode = "rgb",
        class_mode = None,
        batch_size = 1,
        shuffle = False
    )

    best_model = load_model(os.path.join(checkpoint_filepath, model))
    generator.reset()
    preds = best_model.predict(
        generator,
        verbose = 1
    )

    results = pd.DataFrame({
        "Filename": generator.filenames,
        "Prediction": np.argmax(preds, axis=-1)
    })
    return results

def run(filename):
    convert_video_to_image(filename)
    crop_faces(filename)
    results = predict(filename)
    return results


if __name__ == "__main__":
    filename = "real.mp4"
    print(run(filename))
