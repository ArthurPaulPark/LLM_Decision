import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("="*60)
print(f"[INFO] 모델 사전 캐싱(Pre-caching) 시작: {MODEL_NAME}")
print("="*60)
print("이 작업은 모델의 크기에 따라 수 분 이상 소요될 수 있습니다.")
print("인터넷이 연결된 마스터 노드에서 실행해야 정상적으로 다운로드됩니다.\n")

print("[1/3] Tokenizer 다운로드 및 캐싱 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("✓ Tokenizer 캐싱 완료!\n")

print("[2/3] Model 가중치 다운로드 및 캐싱 중...")
# 모델을 메모리에 통째로 로드하는 대신 snapshot_download를 사용하여 안전하고 빠르게 캐싱
snapshot_download(repo_id=MODEL_NAME, ignore_patterns=["*.gguf", "*.h5", "*.msgpack"])
print("✓ Model 가중치 캐싱 완료!\n")

print("="*60)
print("[INFO] 사전 캐싱이 모두 완료되었습니다!")
print("[INFO] 이제 sbatch submit_qwen.sh 명령어를 통해 계산 노드로 작업을 던져주세요.")
print("="*60)
