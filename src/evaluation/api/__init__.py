from src.evaluation.api.gemini import GeminiClient
from src.evaluation.api.gpt import GPTClient
from src.evaluation.api.vllm import VllmClient


CLIENT_MAP = {
    'gemini': GeminiClient,
    'gpt': GPTClient,
    'vllm': VllmClient,
}