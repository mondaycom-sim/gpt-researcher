"""Amazon Bedrock provider for GPT-Researcher.

Alternative LLM provider that routes research queries through
AWS Bedrock instead of OpenAI or Anthropic direct APIs.
Supports both Claude and Titan model families.
"""
import json
from typing import Optional

import boto3
from botocore.exceptions import ClientError

BEDROCK_MODELS = {
    "claude-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "titan-express": "amazon.titan-text-express-v1",
    "titan-lite": "amazon.titan-text-lite-v1",
}


class BedrockProvider:
    """Bedrock foundation model provider for research tasks."""

    def __init__(self, region: str = "us-east-1", model_key: str = "claude-sonnet"):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = BEDROCK_MODELS.get(model_key, model_key)
        self._is_anthropic = "anthropic" in self.model_id

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        system: Optional[str] = None,
    ) -> str:
        """Generate a research completion via Bedrock."""
        if self._is_anthropic:
            return self._invoke_anthropic(prompt, max_tokens, temperature, system)
        return self._invoke_titan(prompt, max_tokens, temperature)

    def _invoke_anthropic(self, prompt, max_tokens, temperature, system=None):
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["content"][0]["text"]

    def _invoke_titan(self, prompt, max_tokens, temperature):
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": 0.9,
            },
        }
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        return result["results"][0]["outputText"]

    def health_check(self) -> bool:
        """Verify Bedrock connectivity and model access."""
        try:
            bedrock = boto3.client("bedrock", region_name="us-east-1")
            bedrock.get_foundation_model(modelIdentifier=self.model_id)
            return True
        except ClientError:
            return False