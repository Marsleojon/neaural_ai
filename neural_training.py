import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


##############################
# Static Variable Initialization
##############################

path = 'data'
test_ratio = 0.2
validation_ratio = 0.2
image_dimension = (32, 32, 3)
batch_size_val = 32
epochs_val = 100

##############################
# Dynamic Variable Initialization
##############################

images = []
class_no = []
my_list = os.listdir(path)
no_of_classes = len(my_list)
num_of_samples = []
##############################

for x in range(0, no_of_classes):
    my_pic_list = os.path.join(path, str(x))
    for img in os.listdir(my_pic_list):
        try:
            cur_img = cv2.imread(os.path.join(
                my_pic_list, img), cv2.IMREAD_COLOR)
            cur_img = cv2.resize(
                cur_img, (image_dimension[0], image_dimension[1]))
            images.append(cur_img)
            class_no.append(x)
        except Exception:
            pass
# print(len(class_no))
images = np.array(images)
class_no = np.array(class_no)
print(images.shape)

# Splitting the data

x_train, x_test, y_train, y_test = train_test_split(
    images, class_no, test_size=test_ratio)
x_train, x_validation, y_train, y_validation = train_test_split(
    x_train, y_train, test_size=validation_ratio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

for x in range(0, no_of_classes):
    num_of_samples.append(len(np.where(y_train == x)[0]))
print(num_of_samples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, no_of_classes), num_of_samples)
plt.title("No. of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


def pre_processing(img_value):
    img = cv2.cvtColor(img_value, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img


x_train = np.array(list(map(pre_processing, x_train)))
x_test = np.array(list(map(pre_processing, x_test)))
x_validation = np.array(list(map(pre_processing, x_validation)))

x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_validation = x_validation.reshape(
    x_validation.shape[0], x_validation.shape[1], x_validation.shape[2], 1)

data_generator = ImageDataGenerator(width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    zoom_range=0.2,
                                    shear_range=0.1,
                                    rotation_range=10)
data_generator.fit(x_train)

y_train = to_categorical(y_train, no_of_classes)
y_test = to_categorical(y_test, no_of_classes)
y_validation = to_categorical(y_validation, no_of_classes)


def action_returned_model():
    no_of_filters = 60
    size_of_filter1 = (5, 5)
    size_of_filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_of_node = 500

    model = Sequential()
    model.add((Conv2D(no_of_filters, size_of_filter1, input_shape=(image_dimension[0],
                                                                   image_dimension[1],
                                                                   1), activation='relu')))
    model.add((Conv2D(no_of_filters, size_of_filter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(no_of_filters//2, size_of_filter2, activation='relu')))
    model.add((Conv2D(no_of_filters//2, size_of_filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_of_node, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(no_of_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


model = action_returned_model()

print(model.summary())

history = model.fit_generator(data_generator.flow(x_train, y_train,
                                                  batch_size=32),
                              validation_data=(x_validation, y_validation),
                              steps_per_epoch=len(x_train) // 32,
                              epochs=epochs_val,
                              shuffle=True)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy = ', score[1])

model.save("trained_memory/training_model.h5")
