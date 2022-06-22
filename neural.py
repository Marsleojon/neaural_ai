import numpy as np
import cv2
from keras.models import load_model
import imutils
from matplotlib import pyplot as plt
import psycopg2
from psycopg2 import Error
import datetime


# Static Variable
model = load_model('trained_memory/training_model.h5')
threshold = 0.60
image_size = (1600, 1200)
y_point = 180
h_point = 550
x_point = 470
w_point = 1250


img_raw_img = cv2.imread('metersample2.jpg')

img_raw_img = cv2.resize(img_raw_img, image_size)

raw_img = img_raw_img[y_point:h_point, x_point:w_point]


gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction

edged = cv2.Canny(bfilter, 30, 50)

plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))  # Showing Crop image

thresh_inv = cv2.threshold(
    edged, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Blurring the image
blur = cv2.GaussianBlur(thresh_inv, (1, 1), 0)
# Getting threshold value
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# find contours
contours = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Getting mask value using np.ones
mask = np.ones(edged.shape[:2], dtype="uint8") * 255

highest_rectangle = 0
counter = 0
x_list = []
y_list = []
h_list = []
w_list = []
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    if w*h > 1000:
        if (w*h) > highest_rectangle:
            x_list.append(x)
            y_list.append(y)
            h_list.append(h)
            w_list.append(w)
            highest_rectangle = w*h
            counter = counter + 1
print("Counter", counter)
index = 0
highest_area = 0
for i in range(0, len(w_list)):
    temp_area = w_list[i] * h_list[i]
    if temp_area > highest_area:
        index = i
        highest_area = temp_area


# Getting the shape of the rectangle
image = cv2.rectangle(mask, (x_list[index], y_list[index]), (
    x_list[index]+w_list[index], y_list[index]+h_list[index]), (0, 0, 255), -1)

# Showing the block of a square/rectangle only in the image
res_final = cv2.bitwise_and(raw_img, raw_img, mask=cv2.bitwise_not(mask))


# crop the value of the given image using the size of the square that is location in the image
first_raw_image = res_final[y_list[index]:h_list[index] +
                            y_list[index], x_list[index]: x_list[index] + w_list[index]]


first_raw_gray = cv2.cvtColor(first_raw_image, cv2.COLOR_BGR2GRAY)


first_raw_thresh_inv = cv2.threshold(
    first_raw_gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

# Blur the image
first_raw_blur = cv2.GaussianBlur(first_raw_thresh_inv, (1, 1), 0)

first_raw_thresh = cv2.threshold(
    first_raw_blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

# find contours
first_raw_contours = cv2.findContours(
    first_raw_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

first_raw_mask = np.ones(first_raw_image.shape[:2], dtype="uint8") * 255

first_counter_raw = 0
first_cr_x_list = []
first_cr_y_list = []
first_cr_h_list = []
first_cr_w_list = []
for contour in first_raw_contours:
    first_cr_x, first_cr_y, first_cr_w, first_cr_h = cv2.boundingRect(contour)
    if first_cr_w*first_cr_h > 1000:
        first_cr_x_list.append(first_cr_x)
        first_cr_y_list.append(first_cr_y)
        first_cr_h_list.append(first_cr_h)
        first_cr_w_list.append(first_cr_w)
        first_counter_raw = first_counter_raw + 1


for raw in range(0, len(first_cr_x_list)):
    temp_x = first_cr_x_list[raw]
    for i in range(raw, len(first_cr_x_list)):
        if first_cr_x_list[i] < temp_x:
            cr_temp_x = temp_x
            temp_x = first_cr_x_list[i]
            first_cr_x_list[raw] = first_cr_x_list[i]
            first_cr_x_list[i] = cr_temp_x

            cr_temp_y = first_cr_y_list[raw]
            first_cr_y_list[raw] = first_cr_y_list[i]
            first_cr_y_list[i] = cr_temp_y

            cr_temp_h = first_cr_h_list[raw]
            first_cr_h_list[raw] = first_cr_h_list[i]
            first_cr_h_list[i] = cr_temp_h

            cr_temp_w = first_cr_w_list[raw]
            first_cr_w_list[raw] = first_cr_w_list[i]
            first_cr_w_list[i] = cr_temp_w

cr_final_x_list = []
cr_final_y_list = []
cr_final_h_list = []
cr_final_w_list = []
raw_counter = 0

# Sorting the value
for raw in range(0, len(first_cr_x_list)):
    if (first_cr_w_list[raw] * first_cr_h_list[raw]) > 10000 and (first_cr_w_list[raw] * first_cr_h_list[raw]) < 50000:
        print("Area", (first_cr_w_list[raw] * first_cr_h_list[raw]))
        cr_final_x_list.append(first_cr_x_list[raw])
        cr_final_y_list.append(first_cr_y_list[raw])
        cr_final_h_list.append(first_cr_h_list[raw])
        cr_final_w_list.append(first_cr_w_list[raw])
        raw_counter = raw_counter + 1

print("raw_counter", raw_counter)


def pre_processing(img_value):
    img = cv2.cvtColor(img_value, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


prediction_reading = ''
for raw_index in range(0, raw_counter):
    cv2.rectangle(first_raw_mask, (cr_final_x_list[raw_index], cr_final_y_list[raw_index]),
                  (cr_final_x_list[raw_index]+cr_final_w_list[raw_index],
                   cr_final_y_list[raw_index]+cr_final_h_list[raw_index]), (0, 0, 255), -1)

    new_image = cv2.bitwise_and(
        first_raw_image, first_raw_image, mask=cv2.bitwise_not(first_raw_mask))

    cropped_number = new_image[cr_final_y_list[raw_index]:cr_final_h_list[raw_index]+cr_final_y_list[raw_index],
                               cr_final_x_list[raw_index]: cr_final_x_list[raw_index] + cr_final_w_list[raw_index]]

    img = cv2.resize(cropped_number, (32, 32))

    img = pre_processing(img)
    img = img.reshape(1, 32, 32, 1)
    # Prediction of number
    class_index = int(model.predict_classes(img))
    print("class prediction", class_index)
    predictions = model.predict(img)
    probability_value = np.amax(predictions)
    print("probility percentage", probability_value)
    print("probability prediction", class_index)
    prediction_reading = prediction_reading+'' + str(class_index)
print("prediction meter:", prediction_reading)
