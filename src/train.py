from src.model_architecture import Architecture
from keras.optimizers import Adam
from keras.layers import Input
import mlflow
import mlflow.keras

architecture = Architecture()

image_height = 256
image_width = 256
learning_rate = 0.01
epochs = 10
batch_size = 2

class Trainer:

    def model_trainer(self, X_train, X_valid, y_train, y_valid):

        mlflow.set_experiment("My Experiments")

        with mlflow.start_run():

            input_img = Input((image_height, image_width, 1), name='img')
            model = architecture.unet(input_img, n_filters=16, kernel_size=3, dropout=0.01, batchnorm=True)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=["accuracy"])
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid))
            training_accuracy = history.history['accuracy'][-1]
            validation_accuracy = history.history['val_accuracy'][-1]

            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)

            mlflow.log_metric("training_accuracy", training_accuracy)
            mlflow.log_metric("validation_accuracy", validation_accuracy)


            return training_accuracy, validation_accuracy