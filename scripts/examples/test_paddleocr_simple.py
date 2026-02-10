import requests
import base64
import json
import mimetypes
import os

def encode_image(image_source):
	mime_type, _ = mimetypes.guess_type(image_source)
	if mime_type is None:
		print(f"Error: Could not determine MIME type for '{image_source}'")
		return None
	
	# Check if the source is a URL or local file
	if image_source.startswith(('http://', 'https://')):
		response = requests.get(image_source)
		response.raise_for_status()
		image_bytes = response.content
	else:
		with open(image_source, 'rb') as f:
			image_bytes = f.read()
	
	# Encode to base64
	base64_image = base64.b64encode(image_bytes).decode('utf-8')
	return f"data:{mime_type};base64,{base64_image}"

def process_image(image_data_url, api_endpoint, api_key):
	# Prepare payload
	payload = {
		"input": [{
			"type": "image_url",
			"url": image_data_url,
		}]
	}
	
	# Make inference request
	headers = {
		'accept': 'application/json',
		'Content-Type': 'application/json',
		'Authorization': f'Bearer {api_key}'
	}
	
	response = requests.post(api_endpoint, headers=headers, json=payload)
	response.raise_for_status()
	return response.json()

# Process the sample image
image_source = "https://assets.ngc.nvidia.com/products/api-catalog/nemo-retriever/object-detection/page-elements-example-1.jpg"
# Also works with local files
# image_source = "path/to/your/image.jpg"
api_endpoint = "https://ml-9132483a-8f3.gr-docpr.a465-9q4k.cloudera.site/namespaces/serving-default/endpoints/paddle-ocr/v1/infer"

# If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
API_KEY = os.environ["JWT_TOKEN"]


try:
	# Encode the image
	image_data_url = encode_image(image_source)
	
	# Process the image
	result = process_image(image_data_url, api_endpoint, API_KEY)
	print(json.dumps(result, indent=2))
except requests.exceptions.RequestException as e:
	print(f"API request failed: {e}")
except Exception as e:
	print(f"Error: {e}")
