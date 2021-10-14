import tensorflow as tf
import os
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path,getcwd,chdir
zip_ref = zipfile.ZipFile('tmp2/happy-or-sad.zip','r')
zip_ref.extractall('tmp2/h-or-s')
zip_ref.close()
train_happy_dir = os.path.join('tmp2/h-or-s/happy')

# Directory with our training human pictures
train_sad_dir = os.path.join('tmp2/h-or-s/sad')
train_happy_names = os.listdir(train_happy_dir)
train_sad_names = os.listdir(train_sad_dir)
print(train_happy_names[:10])
print(train_sad_names[:10])
print('total training horse images:',len(os.listdir(train_happy_dir)))
print('total training human images:',len(os.listdir(train_sad_dir)))
def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') is not None and logs.get('accuracy') >= DESIRED_ACCURACY):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    callbacks = myCallback()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
    train_datagen = ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        'tmp2/h-or-s',
        target_size=(150,150),
        batch_size=10,
        class_mode='binary'
    )
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        verbose=1,
        callbacks = [callbacks])
    return history.history['accuracy'][-1]
train_happy_sad_model()



