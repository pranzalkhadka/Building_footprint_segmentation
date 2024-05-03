import sys
import requests


class DataIngestion:


    def download_file_from_google_drive(self, file_id, destination):
        URL = "https://docs.google.com/uc?export=download&confirm=1"

        session = requests.Session()

        response = session.get(URL, params={"id": file_id}, stream=True)
        token = self.get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(URL, params=params, stream=True)

        self.save_response_content(response, destination)


    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None


    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

if __name__ == "__main__":
    file_id = "11dL_y2UbChdY6-OFgkyauf9KeXgeN5Cv"
    path = "/home/pranjal/Downloads/Building_footprint_segmentation/Data/segmentation_data.zip"
    DataIngestion().download_file_from_google_drive(file_id, path)
