# import boto3

# bedrock = boto3.client(service_name="bedrock")
# response = bedrock.list_foundation_models(byProvider="anthropic")

# for summary in response["modelSummaries"]:
#    print(summary["modelId"])

import boto3
import json

# https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html#model-parameters-claude
bedrock = boto3.client(service_name="bedrock-runtime")
body = json.dumps(
    {
        "prompt": "\n\nHuman: Tell me a funny joke about outer space\n\nAssistant:",
        "max_tokens_to_sample": 500,
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 250,
    }
)

response = bedrock.invoke_model(body=body, modelId="anthropic.claude-v2")

response_body = json.loads(response.get("body").read())
print(response_body.get("completion"))
print(response_body)
