from src.model_architecture import Architecture
from keras.optimizers import Adam
from keras.layers import Input


architecture = Architecture()

image_height = 256
image_width = 256

class Trainer:

    def model_trainer(self, X_train, X_valid, y_train, y_valid):

        input_img = Input((image_height, image_width, 1), name='img')
        model = architecture.unet(input_img, n_filters=16, kernel_size=3, dropout=0.01, batchnorm=True)
        model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=["accuracy"])
        history = model.fit(X_train, y_train, batch_size=2, epochs=10, validation_data=(X_valid, y_valid))
        training_accuracy = history.history['accuracy'][-1]
        validation_accuracy = history.history['val_accuracy'][-1]

        return training_accuracy, validation_accuracy