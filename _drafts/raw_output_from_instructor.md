```python
import json

import instructor
from pydantic import BaseModel


class Greeting(BaseModel):
    """
    This is included.
    """

    message: str


client = instructor.from_provider("openai/gpt-4.1-mini")


def log_payload(*_, **kwargs):
    # kwargs is the request body after Instructor has transformed your response_model into tools/tool_choice
    print("=== REQUEST PAYLOAD ===")
    print(json.dumps(kwargs, indent=2))


def log_response(resp):
    print("=== RAW RESPONSE ===")
    # for OpenAI, resp is a ChatCompletion dataclass-like object
    print(resp.model_dump_json(indent=2))


client.on("completion:kwargs", log_payload)
client.on("completion:response", log_response)

result = client.chat.completions.create(
    response_model=Greeting,
    messages=[{"role": "user", "content": "Please respond with a JSON object that matches the schema below."}],
)

```
