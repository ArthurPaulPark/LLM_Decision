#!/usr/bin/env python3
"""
HuggingFace Hub 업로드 스크립트.

두 가지 모드:
  adapter : LoRA 어댑터만 업로드 (~100MB, 빠름)
            → 사용 시 base model(Qwen3-4B) + 어댑터 별도 로드 필요
  merged  : LoRA를 base model에 병합 후 전체 업로드 (~8GB, 느림)
            → 사용 시 일반 HF 모델처럼 바로 로드 가능

⚠ compute 노드는 인터넷 차단 → 반드시 로그인 노드(storage)에서 실행

사용법:
    # 어댑터만 업로드 (권장, 빠름)
    python upload_to_hub.py --repo username/my-model --token hf_xxxx

    # 병합 모델 업로드 (편리하지만 느림, RAM 16GB+ 필요)
    python upload_to_hub.py --repo username/my-model --token hf_xxxx --mode merged

    # 토큰을 환경변수로 설정하는 경우
    export HF_TOKEN=hf_xxxx
    python upload_to_hub.py --repo username/my-model
"""

import os
import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── 모드 1: 어댑터만 업로드 ───────────────────────────────────────────────────
def upload_adapter_only(
    adapter_path: str,
    repo_id: str,
    token: str,
    private: bool = True,
    commit_message: str = "Upload QLoRA adapter",
) -> None:
    """
    LoRA 어댑터 파일만 Hub에 업로드.
    모델 로드 없이 파일만 전송 → 빠르고 가벼움.

    다른 PC에서 사용법:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base, "<repo_id>")
        tokenizer = AutoTokenizer.from_pretrained("<repo_id>")
    """
    from huggingface_hub import HfApi

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    api = HfApi(token=token)

    # 레포지토리 생성 (이미 있으면 무시)
    logger.info(f"Creating/verifying repo: {repo_id}")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # 어댑터 폴더 전체 업로드
    logger.info(f"Uploading adapter from {adapter_path} → {repo_id} ...")
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        commit_message=commit_message,
        ignore_patterns=["*.log", "*.tmp", "__pycache__/*"],
    )

    logger.info(f"✓ 업로드 완료: https://huggingface.co/{repo_id}")
    _print_usage_adapter(repo_id)


# ── 모드 2: 병합 모델 업로드 ─────────────────────────────────────────────────
def upload_merged_model(
    base_model: str,
    adapter_path: str,
    repo_id: str,
    token: str,
    private: bool = True,
) -> None:
    """
    LoRA를 base model에 병합(merge_and_unload) 후 전체 모델 업로드.
    업로드 후 PEFT 없이 일반 모델처럼 사용 가능.

    ⚠ RAM 16GB+ 필요 (CPU 로드 시 fp32 ~16GB)
    ⚠ 업로드 시간: 8GB 모델 기준 네트워크 속도에 따라 수십 분 소요

    다른 PC에서 사용법:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("<repo_id>", torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained("<repo_id>")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # 로그인 노드는 GPU 없을 수 있음 → CPU 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16  # bfloat16: CPU에서도 동작 (RAM ~8GB)

    logger.info(f"Base model 로드 중 ({device}, {dtype}): {base_model}")
    logger.info("(CPU 로드는 수 분 소요될 수 있습니다)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    logger.info(f"LoRA 어댑터 로드 중: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    logger.info("LoRA 가중치 병합 중 (merge_and_unload)...")
    model = model.merge_and_unload()
    logger.info("✓ 병합 완료")

    # 토크나이저 로드 (어댑터 폴더에 저장된 것 우선)
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Hub 업로드
    logger.info(f"모델 업로드 중 → {repo_id} (시간이 걸립니다...)")
    model.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message="Upload merged Qwen3-4B + QLoRA model",
    )
    tokenizer.push_to_hub(
        repo_id,
        token=token,
        private=private,
        commit_message="Upload tokenizer",
    )

    logger.info(f"✓ 업로드 완료: https://huggingface.co/{repo_id}")
    _print_usage_merged(repo_id)


# ── 사용법 출력 ───────────────────────────────────────────────────────────────
def _print_usage_adapter(repo_id: str) -> None:
    print("\n" + "=" * 60)
    print(" 다른 PC에서 사용하는 방법 (어댑터 모드)")
    print("=" * 60)
    print(f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTER    = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()
""")
    print("=" * 60)


def _print_usage_merged(repo_id: str) -> None:
    print("\n" + "=" * 60)
    print(" 다른 PC에서 사용하는 방법 (병합 모드)")
    print("=" * 60)
    print(f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    REPO,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
""")
    print("=" * 60)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuned 모델을 HuggingFace Hub에 업로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HF 레포지토리 ID (예: myusername/qwen3-4b-coding-agent)",
    )
    parser.add_argument(
        "--base_model",
        default="google/gemma-4-E2B-it",
        help="Base 모델명 (default: google/gemma-4-E2B-it)",
    )
    parser.add_argument(
        "--adapter",
        default="outputs/adapter_package",
        help="LoRA 어댑터 경로 (default: outputs/adapter_package, quantize.py 실행 후)",
    )
    parser.add_argument(
        "--mode",
        choices=["adapter", "merged"],
        default="adapter",
        help="adapter: 어댑터만 업로드(~100MB) | merged: 전체 모델 업로드(~8GB) (default: adapter)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace 토큰 (없으면 HF_TOKEN 환경변수 사용)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="공개 레포로 업로드 (기본: private)",
    )
    args = parser.parse_args()

    # 토큰 확인
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace 토큰이 필요합니다.\n"
            "  방법 1: --token hf_xxxx\n"
            "  방법 2: export HF_TOKEN=hf_xxxx"
        )

    private = not args.public

    logger.info(f"모드   : {args.mode}")
    logger.info(f"레포   : {args.repo} ({'private' if private else 'public'})")
    logger.info(f"어댑터 : {args.adapter}")

    if args.mode == "adapter":
        upload_adapter_only(args.adapter, args.repo, token, private)
    else:
        upload_merged_model(args.base_model, args.adapter, args.repo, token, private)


if __name__ == "__main__":
    main()
