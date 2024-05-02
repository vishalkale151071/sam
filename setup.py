"""
This script is used to set up the project for testing segment anything model.
"""
import os
import requests

HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

def download_weights():
    """
    This function is used to download the weights for the model.
    """

    # Check if the weights directory exists
    if not check_weights():
        print("Downloading weights")

        # Define the URL of the file to download
        WEIGHT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

        # Define the path where you want to save the file
        save_path = os.path.join(HOME, "weights")

        # Make sure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Send an HTTP GET request to download the file
        response = requests.get(WEIGHT_URL, timeout=10)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the downloaded content to the specified path
            with open(save_path + '/sam_vit_h_4b8939.pth', "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Failed to download the file.")


def check_weights():
    """
    This function is used to check if the weights are downloaded.
    """
    if os.path.isfile(CHECKPOINT_PATH):
        print("Weights exists")
        return True
    print("Weights does not exist")
    return False

def check_create_data():
    """
    This function is used to check if the data is downloaded.
    """
    data_path = os.path.join(HOME, "data")
    if os.path.exists(data_path):
        print("Data exists")
        return
    print("Data does not exist")
    os.makedirs(data_path, exist_ok=True)

if __name__ == "__main__":
    download_weights()
    check_create_data()
    