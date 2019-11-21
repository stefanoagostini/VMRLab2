from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K

N_SAMPLES_TRAIN = 498
VAL_SPLIT = 0.2
N_TRAIN_SAMPLES = int(N_SAMPLES_TRAIN * (1-VAL_SPLIT) +1)
N_VAL_SAMPLES = int(N_SAMPLES_TRAIN * VAL_SPLIT)
IMAGE_WIDTH = IMAGE_HEIGHT = 256
BATCH_SIZE = 20

print("[INFO] Loading data...")

train_dir = "dataset/train"
test_dir = "dataset/test"

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), shuffle=True,
                                                    seed=13, batch_size=BATCH_SIZE, class_mode='binary', subset="training")
val_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), shuffle=True,
                                                  seed=13, batch_size=BATCH_SIZE, class_mode='binary', subset="validation")

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), shuffle=False,
                                                  seed=13, batch_size=BATCH_SIZE, class_mode='binary')

K.clear_session()

print("[INFO] Compiling model...")

shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
base_model = MobileNetV2(input_shape=shape, weights='imagenet', include_top=False)
for i in range(len(base_model.layers)):
    base_model.layers[i].trainable = False
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)
model = Model(inputs=base_model.input, outputs=head_model)
print(model.summary())

opt = RMSprop(lr=1e-4)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model...")

model.fit_generator(train_generator, epochs=20, steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
                    validation_data=val_generator, validation_steps=N_VAL_SAMPLES // BATCH_SIZE, verbose=2)

model.save('MobileNetV2.h5')
