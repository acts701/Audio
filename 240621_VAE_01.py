from tensorflow.keras import datasets
import numpy as np
(x_train, y_train), (x_text, y_test) = datasets.fashion_mnist.load_data()

def preprocess(imgs):
    imgs = imgs.astype("float32") / 255.0
    imgs = np.pad(imgs, ((0,0), (2,2), (2,2)), constant_values=0.0)
    imgs = np.expand_dims(imgs, -1)
    return imgs
x_train = preprocess(x_train)
x_test = preprocess(x_test)

encoder_input = layers.Input(shape=(32,32,1), name = "encoder_input")
x = layers.Conv2D(32,(3,3)), strides = 2, activation = 'relu', padding = 'same')(encoder_input)