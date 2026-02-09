import os
from openai import OpenAI
import json


# If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
API_KEY = os.environ["JWT_TOKEN"]

MODEL_ID = "reducto/RolmOCR"

client = OpenAI(
	base_url="https://ml-9132483a-8f3.gr-docpr.a465-9q4k.cloudera.site/namespaces/serving-default/endpoints/rolmocr/openai/v1",
	api_key=API_KEY,
)

completion = client.chat.completions.create(
	model=MODEL_ID,
	messages=[{"role": "user", "content": "Write a one-sentence definition of GenAI."}],
	temperature=0.2,
	top_p=0.7,
	max_tokens=1024,
	stream=True
)

for chunk in completion:
	if chunk.choices[0].delta.content is not None:
		print(chunk.choices[0].delta.content, end="")