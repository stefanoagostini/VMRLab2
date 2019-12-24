import argparse
from keras.models import load_model
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import time


IMAGE_WIDTH = IMAGE_HEIGHT = 256

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="myNet1_0.001.hdf5")
args = vars(ap.parse_args())
model_path = args["model"]

print("[INFO] Loading data...")
test_dir = "dataset/test"
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), shuffle=False,
                                                  seed=13, batch_size=1, class_mode='binary')
filenames = test_generator.filenames
nb_samples = len(filenames)

print("[INFO] Loading model...")
K.clear_session()
model = load_model(model_path)

print("[INFO] Predicting...")
start = time.time()
predictions = model.predict_generator(test_generator, steps=nb_samples, verbose=0)
end = time.time()

print("[INFO] Computing results...")
avgtime = (end-start)/nb_samples
print("Average inference time: {0:.5f}".format(avgtime))
fps = 1/avgtime
print("FPS: {}".format(int(fps)))
predicted_classes = predictions
predicted_classes[predicted_classes <= 0.5] = 0
predicted_classes[predicted_classes > 0.5] = 1
print(test_generator.class_indices)
print(model.evaluate_generator(test_generator, steps=test_generator.samples))
report = classification_report(test_generator.classes, predicted_classes, target_names=test_generator.class_indices)
print(report)
confusion = confusion_matrix(test_generator.classes, predicted_classes)
print(confusion)

# Confusion matrix with tensor flow
# import tensorflow
# c = tensorflow.math.confusion_matrix(labels=test_generator.classes, predictions=predicted_classes)
# print(c.eval(session=tensorflow.compat.v1.Session()))
