"""Small wrapper around Roboflow's hosted inference API."""
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

API_URL = "https://serverless.roboflow.com"


def create_client():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    model_id = os.getenv("ROBOFLOW_MODEL_ID")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Add it to your .env file.")
    if not model_id:
        raise RuntimeError("ROBOFLOW_MODEL_ID is missing. Add it to your .env file.")
    return InferenceHTTPClient(api_url=API_URL, api_key=api_key), model_id


def infer_image(image_path: str):
    client, model_id = create_client()
    return client.infer(image_path, model_id=model_id)
