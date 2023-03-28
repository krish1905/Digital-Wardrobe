# utils.py
# (c) Krish Garg 2023
# Digital Wardrobe
# This file contains all the other functions, so it is easy to call them in the digitalWardrobe.py file when needed
import cv2
# import matplotlib.pyplot as plt
from yolov3_tf2.models import YoloV3
import numpy as np
import time
import os
import tensorflow as tf
from bs4 import BeautifulSoup
import requests
import random
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

def boundingBoxAndCrop(img, list_obj):
    try:
        # Convert tensor to numpy array
        img = img.numpy()
    except:
        pass

    # Remove singleton dimensions
    img = np.squeeze(img)

    # Get image height and width
    img_width = img.shape[1]
    img_height = img.shape[0]

    # Define colors for bounding boxes
    colors = [[244 / 255, 241 / 255, 66 / 255], [66 / 255, 241 / 255, 66 / 255], [241 / 255, 66 / 255, 66 / 255],
              [255 / 255, 99 / 255, 71 / 255], [255 / 255, 165 / 255, 0 / 255]]
    colors_count = 0

    labels = []
    cropped_img_list = []
    # Draw rectangle bounding box
    for i, obj in enumerate(list_obj):
        x1 = int(round(obj['x1']*img_width))
        y1 = int(round(obj['y1']*img_height))
        x2 = int(round(obj['x2']*img_width))
        y2 = int(round(obj['y2']*img_height))

        color = colors[colors_count % len(colors)]
        colors_count += 1

        text = '{}: {:.2f}'.format(obj['label'], obj['confidence'])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        img = cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # Crop clothing images using bounding boxes
        cropped_img = img[y1:y2, x1:x2, :]
        cropped_img_list.append(cropped_img)
        labels.append(obj['label'])

    for i, cropped_img in enumerate(cropped_img_list):
        # to view cropped file: (uncomment if errors with saved cropped images)
        '''
        plt.figure(figsize=(3, 3))
        plt.imshow(cropped_img)
        plt.axis('off')
        plt.title(f'Cropped Image {i + 1}')
        plt.show()
        '''
        label = labels[i]
        label_dir = os.path.join("digitalWardrobe", label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        num_files = len(os.listdir(label_dir)) + 1
        filename = os.path.join(label_dir, f"{num_files}.jpeg")
        # Convert image file to colored
        img_color = cropped_img * 255
        # Save cropped image to file
        cv2.imwrite(filename, img_color)

    return img

def imgToTensor(img_path):
    # Read image file from the provided path
    img_raw = tf.io.read_file(img_path)
    # Decode the image raw data into a tensor
    img = tf.image.decode_image(img_raw, channels=3, dtype=tf.dtypes.float32)
    # Expand the dimensions of the image to fake a batch axis, required by some TensorFlow functions
    img = tf.expand_dims(img, 0)

    return img

def deepfashion2_yolov3():
    t1 = time.time()
    model = YoloV3(classes=13)
    model.load_weights('./built_model/deepfashion2_yolov3')
    t2 = time.time()
    print('Load DeepFashion2 Yolo-v3 from disk: {:.2f} sec'.format(t2 - t1))

    return model

def checkWeather(city_name):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    city_name = city_name.replace(" ", "+")
    try:
        res = requests.get(
            f'https://www.google.com/search?q={city_name}&oq={city_name}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8',
            headers=headers)

        print("Loading...\n")

        soup = BeautifulSoup(res.text, 'html.parser')
        location = soup.select('#wob_loc')[0].getText().strip()
        time = soup.select('#wob_dts')[0].getText().strip()
        info = soup.select('#wob_dc')[0].getText().strip()
        temperature = soup.select('#wob_tm')[0].getText().strip()

        print("Location: " + location)
        print("Temperature: " + temperature + "°C")
        print("Time: " + time)
        print("Weather Description: " + info)

    except:
        print("Please enter a valid city name")

    return temperature

def chooseClothes(directories):
    all_files = []
    parent_directory = "digitalWardrobe"
    for directory in directories:
        files = [f for f in os.listdir(os.path.join(parent_directory, directory)) if os.path.isfile(os.path.join(parent_directory, directory, f)) and not f.startswith(".")]
        all_files.extend(files)
    chooseClothes = random.choice(all_files)
    chosen_directory = random.choice(directories)
    return chooseClothes, chosen_directory