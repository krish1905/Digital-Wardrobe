# digitalWardrobe.py
# (c) Krish Garg 2023
# Digital Wardrobe
# This project recognizes clothing in images inputted by the user and saves them in a 'Digital Wardrobe'.
# Users can use this to make an outfit for them based on the weather of the city they are living in

import cv2
from utils import imgToTensor, deepfashion2_yolov3, boundingBoxAndCrop, checkWeather, chooseClothes
from cloth_detection import detectClothes
import os
import matplotlib.pyplot as plt
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

print('''Welcome to your digital wardrobe!
1. Make me an outfit. 
2. Add more clothes to your wardrobe.''')
userinput = input("> ")

# If user wants to make an outfit
if userinput == "1":
    # Ask for city and check weather
    city_name = input("\nEnter your city's name: ")
    city_name = city_name + " weather"
    temperature = int(checkWeather(city_name))
    # If temperature is more than 10 degrees, it is warm
    if temperature >= 10:
        print("\nIt feels a bit warm today. Time to get back in your summer clothing!")
        print("Generating outfit...")

        # Choose a top for summer and display
        summer_tops = ['short_sleeve_dress', 'short_sleeve_outwear', 'short_sleeve_top', 'vest']
        summerTop, dir = chooseClothes(summer_tops)
        sTop = cv2.imread(os.path.join("digitalWardrobe", dir, summerTop))
        sTop = cv2.cvtColor(sTop, cv2.COLOR_BGR2RGB)
        plt.imshow(sTop)
        plt.axis("off")
        plt.show()

        # Choose a bottom for summer and display
        summer_bottoms = ['shorts']
        summerBottom, dir2 = chooseClothes(summer_bottoms)
        sBottom = cv2.imread(os.path.join("digitalWardrobe", dir2, summerBottom))
        sBottom = cv2.cvtColor(sBottom, cv2.COLOR_BGR2RGB)
        plt.imshow(sBottom)
        plt.axis("off")
        plt.show()

    # If temperature is more than 10 degrees, it is cold
    elif temperature < 10:
        print("\nIt seems to be cold today. Time to wear some warm clothes")
        print("Generating outfit...")

        # choose a top for winter and display it
        winter_tops = ['long_sleeve_dress', 'long_sleeve_outwear', 'long_sleeve_top']
        winterTop, dir = chooseClothes(winter_tops)
        wTop = cv2.imread(os.path.join("digitalWardrobe", dir, winterTop))
        wTop = cv2.cvtColor(wTop, cv2.COLOR_BGR2RGB)
        plt.imshow(wTop)
        plt.axis("off")
        plt.show()

        # choose a bottom for winter and display it
        winter_bottoms = ['trousers']
        winterBottom, dir = chooseClothes(winter_bottoms)
        wBottom = cv2.imread(os.path.join("digitalWardrobe", dir, winterBottom))
        wBottom = cv2.cvtColor(wBottom, cv2.COLOR_BGR2RGB)
        plt.imshow(wBottom)
        plt.axis("off")
        plt.show()

# If user wants to add clothes to their wardrobe
elif userinput == "2":

    image_file = input("Enter image file name: ")
    if __name__ == '__main__':
        img = imgToTensor('./images/' + image_file)
        model = deepfashion2_yolov3()
        list_obj = detectClothes(img, model)
        img_with_boxes = boundingBoxAndCrop(img, list_obj)
        # Save image with bounding boxes in a folder called 'detect' if there are errors in cropped image
        folder = "./detect/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(folder + "d_" + image_file, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR) * 255)
        print("\nSuccessfully added that piece to your wardrobe!")

        # To view image with bounding boxes
        '''
        cv2.imshow("Clothes Detection", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
