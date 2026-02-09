import os
from openai import OpenAI
import json


# If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
API_KEY = os.environ["JWT_TOKEN"]

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

client = OpenAI(
	base_url="https://ml-c34a298c-ca7.qzhong-1.a465-9q4k.cloudera.site/namespaces/serving-default/endpoints/qwen-endpoint/openai/v1",
	api_key=API_KEY,
)

completion = client.chat.completions.create(
	model=MODEL_ID,
	messages=[{"role": "user", "content": "who are you"}],
	temperature=0.2,
	top_p=0.7,
	max_tokens=1024,
	stream=True
)

for chunk in completion:
	if chunk.choices[0].delta.content is not None:
		print(chunk.choices[0].delta.content, end="")