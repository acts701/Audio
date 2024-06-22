import numpy as np
from tensorflow.keras import datasets, utils
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

NUM_CLASSES = 10
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = utils.to_categorical(y_train, NUM_CLASSES)
y_test = utils.to_categorical(y_test, NUM_CLASSES)

# 한 층이 여러 개의 이전 층으로부터 입력을 받을 때 sequential이 적합하지 않음. API를 써야 함
# model = models.Sequential([
#     layers.Flatten(input_shape=(32,32,3)),
#     layers.Dense(200, activation = 'relu'),
#     layers.Dense(150, activation = 'relu'),
#     layers.Dense(10, activation = 'softmax'),
# ])

# API로 MLP 만들기
# size와 channel(RGB 3개)를 지정하고 배치 크기는 지정하지 않았음. 
# Input 층에 임의의 이미지 개수를 전달할 수 있기 떄문에 배치 크기는 필요하지 않음
input_layer = layers.Input(shape=(32,32,3))
# 입력을 하나의 벡터로 펼친다. 크기는 32 * 32 * 3 = 3,072
# Dense에서 flatten input을 받기 때문에 처리해준다. 
# 유닛의 출력은 이전 층에서 받은 입력과 가중치를 곱하여 더한 것.
# 더하는 것은 상수 항에 해당하는 편향
x = layers.Flatten()(input_layer)
# 활성화 함수는 RElu, LeakyReLU 등
x = layers.Dense(units=200, activation='relu')(x)
x = layers.Dense(units=150, activation='relu')(x)
# 전체 출력 합이 1이 되어야 할 때 softmax
output_layer = layers.Dense(units=10, activation = 'softmax')(x)
model = models.Model(input_layer, output_layer)

opt = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size = 32, epochs = 10, shuffle = True)

model.evaluate(x_test, y_test) 

CLASSES = np.array(['airplane','automobile','bird','cat','deer','dog','forg','horse','ship','truck'])
preds = model.predict(x_test)
preds_single = CLASSES[np.argmax(preds, axis = -1)]
actual_single = CLASSES[np.argmax(y_test, axis = -1)]

n_to_show = 10
indices = np.random.choice(range(len(x_test)), n_to_show)

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i, idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1, n_to_show, i + 1)
    ax.axis('off')
    ax.text(0.5, -0.35, 'pred = ' + str(preds_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.text(0.5, -0.7, 'act = ' + str(actual_single[idx]), fontsize=10, ha='center', transform=ax.transAxes)
    ax.imshow(img)
