import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

model_id = "arn:aws:bedrock:eu-north-1:169133351222:application-inference-profile/trvimnlz84pg"

response = client.converse(
    modelId=model_id,
    messages=[
        {
            "role": "user",
            "content": [
                {"text": "Explain how retail shelf stock detection works."}
            ],
        }
    ],
    inferenceConfig={
        "maxTokens": 200,
        "temperature": 0.5
    }
)

print(response["output"]["message"]["content"][0]["text"])