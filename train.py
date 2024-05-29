from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.trainer import Trainer


file_id = "11dL_y2UbChdY6-OFgkyauf9KeXgeN5Cv"
path = "/home/pranjal/Downloads/Building_footprint_segmentation/Data/segmentation_data.zip"


if __name__ == "__main__":
    DataIngestion().download_file_from_google_drive(file_id, path)
    X_train, X_valid, y_train, y_valid = DataTransformation().data_loader(path)
    training_accuracy, validation_accuracy = Trainer().model_trainer(X_train, X_valid, y_train, y_valid)
    print(f"Training accuracy : {training_accuracy}")
    print(f"Validation accuracy : {validation_accuracy}")