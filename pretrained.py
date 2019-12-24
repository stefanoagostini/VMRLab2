from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2, ResNet50
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import argparse
import sys

# Constant values
N_SAMPLES_TRAIN = 498
VAL_SPLIT = 0.2
N_TRAIN_SAMPLES = int(N_SAMPLES_TRAIN * (1 - VAL_SPLIT) + 1)
N_VAL_SAMPLES = int(N_SAMPLES_TRAIN * VAL_SPLIT)
IMAGE_WIDTH = IMAGE_HEIGHT = 256
BATCH_SIZE = 10

# Argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", choices=["MobileNetV2", "ResNet50"], default="MobileNetV2")
args = vars(ap.parse_args())
model_type = args["model"]

print("[INFO] Loading data...")

train_dir = "dataset/train"

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VAL_SPLIT)
#train_datagen = ImageDataGenerator(rescale=1. / 255, rotation_range=45, zoom_range=0.15, width_shift_range=0.2,
#                                   height_shift_range=0.2, horizontal_flip=True, validation_split=VAL_SPLIT)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), shuffle=True,
                                                    seed=13, batch_size=BATCH_SIZE, class_mode='binary',
                                                    subset="training")
val_generator = train_datagen.flow_from_directory(train_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), shuffle=True,
                                                  seed=13, batch_size=BATCH_SIZE, class_mode='binary',
                                                  subset="validation")

K.clear_session()

print("[INFO] Compiling model...")
# Model definition
shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
if model_type == "MobileNetV2":
    base_model = MobileNetV2(input_shape=shape, weights='imagenet', include_top=False)
elif model_type == "ResNet50":
    base_model = ResNet50(input_shape=shape, weights='imagenet', include_top=False)
else:
    sys.exit("Error in model name")
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(4, 4))(head_model)
head_model = Flatten(name="flatten")(head_model)
head_model = Dense(128, activation="relu")(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(1, activation="sigmoid")(head_model)
model = Model(inputs=base_model.input, outputs=head_model)
print(model.summary())

# Callbacks definitions
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-9)
early_stop = EarlyStopping(monitor="val_loss", mode='min', verbose=1, patience=20)
if model_type == "MobileNetV2":
    mcp_save = ModelCheckpoint('MobileNetV2_wts.hdf5', save_best_only=True, monitor="val_loss", mode='min')
elif model_type == "ResNet50":
    mcp_save = ModelCheckpoint('ResNet50_wts.hdf5', save_best_only=True, monitor="val_loss", mode='min')
tb = TensorBoard(log_dir="logs")
callbacks_list = [early_stop, mcp_save, lr_scheduler, tb]

# Optimizer definition
opt = Adam(lr=1e-5)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training model...")
history = model.fit_generator(train_generator, epochs=100, steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
                              validation_data=val_generator, validation_steps=N_VAL_SAMPLES // BATCH_SIZE, verbose=2,
                              callbacks=callbacks_list)

# Best accuracy
[best_loss, best_ep] = [np.min(history.history["val_loss"]), np.argmin(history.history["val_loss"])]

print("Best loss: {:.4f}  Epoch: {}".format(best_loss, best_ep))
