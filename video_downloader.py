import requests

def download_video(cloudinary_url, save_path="input_video.mp4"):
    response = requests.get(cloudinary_url, stream=True)
    if response.status_code != 200:
        raise Exception("Failed to download video")

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    return save_path
