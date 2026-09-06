#!/usr/bin/env python3
import sys
# torchvision Mocking (transforms, functional, io) to bypass 구버전 torchvision 0.2.0 import bug
from types import ModuleType

# 1. torchvision.transforms.functional.pil_to_tensor Mocking
try:
    import torchvision.transforms.functional as tv_func
    if not hasattr(tv_func, "pil_to_tensor"):
        tv_func.pil_to_tensor = lambda *args, **kwargs: None
except ImportError:
    func_mock = ModuleType("torchvision.transforms.functional")
    func_mock.pil_to_tensor = lambda *args, **kwargs: None
    sys.modules["torchvision.transforms.functional"] = func_mock

# 2. torchvision.transforms.InterpolationMode Mocking
try:
    import torchvision.transforms as tv_trans
    if not hasattr(tv_trans, "InterpolationMode"):
        class MetaInterpolationMode(type):
            def __getattr__(cls, name):
                return name.lower()
        class DummyInterpolationMode(metaclass=MetaInterpolationMode):
            NEAREST = "nearest"
            BILINEAR = "bilinear"
            BICUBIC = "bicubic"
            NEAREST_EXACT = "nearest_exact"
        tv_trans.InterpolationMode = DummyInterpolationMode
except ImportError:
    trans_mock = ModuleType("torchvision.transforms")
    class MetaInterpolationMode(type):
        def __getattr__(cls, name):
            return name.lower()
    class DummyInterpolationMode(metaclass=MetaInterpolationMode):
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        NEAREST_EXACT = "nearest_exact"
    trans_mock.InterpolationMode = DummyInterpolationMode
    sys.modules["torchvision.transforms"] = trans_mock

# 3. torchvision.io (ImageReadMode, decode_image) Mocking
try:
    from torchvision.io import ImageReadMode, decode_image
except ImportError:
    import torchvision
    io_mock = ModuleType("torchvision.io")
    io_mock.ImageReadMode = object
    io_mock.decode_image = lambda *args, **kwargs: None
    sys.modules["torchvision.io"] = io_mock
    if not hasattr(torchvision, "io"):
        torchvision.io = io_mock

# 4. torchvision.transforms.v2 & v2.functional Mocking
try:
    import torchvision.transforms.v2
except ImportError:
    v2_mock = ModuleType("torchvision.transforms.v2")
    sys.modules["torchvision.transforms.v2"] = v2_mock
    
    v2_func_mock = ModuleType("torchvision.transforms.v2.functional")
    sys.modules["torchvision.transforms.v2.functional"] = v2_func_mock
    
    v2_mock.functional = v2_func_mock
"""
GGUF IQ2_XXS 양자화 파이프라인

단계:
  1. LoRA 어댑터 → base model에 병합 (BF16 safetensors)
  2. BF16 safetensors → GGUF F16 변환 (llama.cpp convert_hf_to_gguf.py)
  3. train.jsonl에서 캘리브레이션 텍스트 추출
  4. llama-imatrix로 importance matrix 생성
  5. llama-quantize로 IQ2_XXS 양자화 (~620MB)

사전 준비:
  bash scripts/setup_llamacpp.sh   ← llama.cpp 빌드 (최초 1회)

실행 (로그인 노드에서, GPU 불필요):
  python quantize.py
  python quantize.py --adapter outputs/checkpoints --output outputs/quantized

최종 파일:
  outputs/quantized/model_iq2xxs.gguf  (~620MB)
"""

import os
import json
import subprocess
import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
LLAMACPP_DIR = Path("./llama.cpp")          # setup_llamacpp.sh가 클론하는 위치


# ── Step 1: LoRA 병합 ────────────────────────────────────────────────────────
def merge_lora(base_model: str, adapter_path: str, output_path: str) -> None:
    """
    base model + LoRA → BF16 병합 모델 저장.
    로그인 노드는 GPU 없음 → CPU 로드 (RAM ~3GB 필요).
    """
    logger.info(f"[1/5] LoRA 병합 시작 (CPU, BF16)...")
    logger.info(f"      Base   : {base_model}")
    logger.info(f"      Adapter: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path, safe_serialization=True)

    # 토크나이저는 base model에서 저장 (어댑터 폴더 것도 동일)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    size_gb = sum(
        f.stat().st_size for f in Path(output_path).rglob("*.safetensors")
    ) / 1024**3
    logger.info(f"      ✓ 병합 완료 → {output_path} ({size_gb:.2f} GB)")


# ── Step 2: GGUF F16 변환 ────────────────────────────────────────────────────
def convert_to_gguf_f16(merged_path: str, gguf_f16_path: str) -> None:
    """
    HF safetensors → GGUF F16.
    llama.cpp의 convert_hf_to_gguf.py 사용.
    """
    logger.info(f"[2/5] GGUF F16 변환 중...")

    convert_script = LLAMACPP_DIR / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"{convert_script} not found.\n"
            "bash scripts/setup_llamacpp.sh 을 먼저 실행하세요."
        )

    subprocess.run(
        [
            "python3", str(convert_script),
            str(merged_path),
            "--outfile", str(gguf_f16_path),
            "--outtype", "f16",
        ],
        check=True,
    )
    size_gb = Path(gguf_f16_path).stat().st_size / 1024**3
    logger.info(f"      ✓ GGUF F16 변환 완료 ({size_gb:.2f} GB)")


# ── Step 3: 캘리브레이션 데이터 추출 ─────────────────────────────────────────
def prepare_calibration_data(
    train_jsonl: str,
    output_txt: str,
    num_samples: int = 1000,
) -> None:
    """
    train.jsonl에서 current_prompt + 최근 대화를 추출해 plain text 파일 생성.
    imatrix 생성 시 도메인에 맞는 캘리브레이션 데이터를 쓰면 품질이 높아짐.
    """
    logger.info(f"[3/5] 캘리브레이션 데이터 추출 중 ({num_samples}개)...")

    lines = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line.strip())
            parts = []

            # 최근 히스토리 (3턴)
            for turn in item.get("history", [])[-3:]:
                if turn.get("role") == "user":
                    parts.append(turn.get("content", "")[:200])
                elif turn.get("role") == "assistant_action":
                    parts.append(turn.get("name", ""))

            # 현재 프롬프트
            parts.append(item.get("current_prompt", ""))
            lines.append(" ".join(parts).strip())

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"      ✓ 캘리브레이션 데이터 → {output_txt} ({len(lines)}줄)")


# ── Step 4: importance matrix 생성 ───────────────────────────────────────────
def generate_imatrix(
    gguf_f16_path: str,
    calib_txt_path: str,
    imatrix_path: str,
) -> None:
    """
    llama-imatrix: 캘리브레이션 데이터로 각 가중치의 중요도 계산.
    IQ 양자화에서 중요한 가중치를 더 높은 비트로 보존.
    """
    logger.info(f"[4/5] importance matrix 생성 중 (수 분 소요)...")

    imatrix_bin = LLAMACPP_DIR / "build/bin/llama-imatrix"
    if not imatrix_bin.exists():
        raise FileNotFoundError(
            f"{imatrix_bin} not found.\n"
            "bash scripts/setup_llamacpp.sh 을 먼저 실행하세요."
        )

    subprocess.run(
        [
            str(imatrix_bin),
            "-m", str(gguf_f16_path),
            "-f", str(calib_txt_path),
            "-o", str(imatrix_path),
            "-c", "512",        # context size (캘리브레이션 청크당)
            "--chunks", "100",  # 사용할 청크 수
            "-ngl", "0",        # CPU only (로그인 노드)
        ],
        check=True,
    )
    logger.info(f"      ✓ imatrix → {imatrix_path}")


# ── Step 5: IQ 양자화 (Imatrix 활용) ───────────────────────────────────────────────────
def quantize_iq(
    gguf_f16_path: str,
    gguf_out_path: str,
    imatrix_path: str,
    quant_type: str,
) -> None:
    """
    llama-quantize: IQ 계열(IQ2_XXS, IQ3_S 등) 적용.
    importance matrix 덕분에 동일 용량 K-quant보다 품질이 좋음.
    """
    logger.info(f"[5/5] {quant_type} 양자화 중 (수 분 소요)...")

    quantize_bin = LLAMACPP_DIR / "build/bin/llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"{quantize_bin} not found.\n"
            "bash scripts/setup_llamacpp.sh 을 먼저 실행하세요."
        )

    subprocess.run(
        [
            str(quantize_bin),
            "--imatrix", str(imatrix_path),
            str(gguf_f16_path),
            str(gguf_out_path),
            quant_type,
        ],
        check=True,
    )

    size_mb = Path(gguf_iq2_path).stat().st_size / 1024**2
    logger.info(f"      ✓ IQ2_XXS 완료 → {gguf_iq2_path} ({size_mb:.0f} MB)")

    if size_mb < 1024:
        logger.info(f"      ✅ 1GB 미만 조건 충족! ({size_mb:.0f} MB)")
    else:
        logger.warning(f"      ⚠ 예상보다 큼: {size_mb:.0f} MB")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GGUF IQ2_XXS 양자화 파이프라인")
    parser.add_argument("--base_model",  default=BASE_MODEL,
                        help=f"Base 모델명 (default: {BASE_MODEL})")
    parser.add_argument("--adapter",     default="outputs/checkpoints",
                        help="LoRA 체크포인트 경로")
    parser.add_argument("--train_jsonl", default="data_alt/train.jsonl",
                        help="캘리브레이션용 학습 데이터")
    parser.add_argument("--output",      default="outputs/quantized",
                        help="출력 디렉토리")
    parser.add_argument("--skip_merge",  action="store_true",
                        help="이미 병합된 모델이 있으면 skip")
    parser.add_argument("--calib_samples", type=int, default=1000,
                        help="캘리브레이션 샘플 수 (default: 1000)")
    parser.add_argument("--quant_type", default="Q3_K_M",
                        choices=["Q3_K_M", "Q3_K_S", "Q4_K_M", "Q4_K_S", "Q5_K_M", "IQ2_XXS"],
                        help="양자화 방식 (default: Q3_K_M). IQ2_XXS는 imatrix 필요.")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    merged_path  = out / "merged_bf16"
    gguf_f16     = out / "model_f16.gguf"
    calib_txt    = out / "calib_data.txt"
    imatrix_path = out / "imatrix.dat"

    # 출력 파일명을 quant_type에 맞게
    quant_suffix = args.quant_type.lower().replace("_", "")  # IQ3_S → iq3s
    gguf_out = out / f"model_{quant_suffix}.gguf"

    needs_imatrix = args.quant_type.startswith("IQ")

    logger.info("=" * 55)
    logger.info(f" GGUF {args.quant_type} 양자화 파이프라인")
    logger.info("=" * 55)

    # Step 1: LoRA 병합
    if args.skip_merge and merged_path.exists():
        logger.info("[1/5] 병합 모델 이미 존재 → 건너뜀")
    else:
        merge_lora(args.base_model, args.adapter, str(merged_path))

    # Step 2: GGUF F16 변환
    if not gguf_f16.exists():
        convert_to_gguf_f16(str(merged_path), str(gguf_f16))
    else:
        logger.info(f"[2/5] GGUF F16 이미 존재 → 건너뜀")

    if needs_imatrix:
        # Step 3: 캘리브레이션 데이터 (IQ 계열만 필요)
        if not calib_txt.exists():
            prepare_calibration_data(args.train_jsonl, str(calib_txt), args.calib_samples)
        else:
            logger.info(f"[3/5] 캘리브레이션 데이터 이미 존재 → 건너뜀")

        # Step 4: imatrix
        if not imatrix_path.exists():
            generate_imatrix(str(gguf_f16), str(calib_txt), str(imatrix_path))
        else:
            logger.info(f"[4/5] imatrix 이미 존재 → 건너뜀")

        # Step 5: IQ 계열 양자화
        quantize_iq(str(gguf_f16), str(gguf_out), str(imatrix_path), args.quant_type)
        size_mb = gguf_out.stat().st_size / 1024**2
        logger.info(f"      ✓ {args.quant_type} 완료 → {gguf_out} ({size_mb:.0f} MB)")
    else:
        # Q3_K_M / Q4_K_M / Q5_K_M: imatrix 불필요, 직접 양자화
        logger.info(f"[3/5] 캘리브레이션 불필요 ({args.quant_type}는 K-quant)")
        logger.info(f"[4/5] imatrix 불필요 ({args.quant_type}는 K-quant)")
        logger.info(f"[5/5] {args.quant_type} 양자화 중...")

        quantize_bin = LLAMACPP_DIR / "build/bin/llama-quantize"
        if not quantize_bin.exists():
            raise FileNotFoundError(
                f"{quantize_bin} not found.\n"
                "bash scripts/setup_llamacpp.sh 을 먼저 실행하세요."
            )

        subprocess.run(
            [str(quantize_bin), str(gguf_f16), str(gguf_out), args.quant_type],
            check=True,
        )
        size_mb = gguf_out.stat().st_size / 1024**2
        logger.info(f"      ✓ {args.quant_type} 완료 → {gguf_out} ({size_mb:.0f} MB)")

    logger.info("\n" + "=" * 55)
    logger.info(" 완료 요약")
    logger.info("=" * 55)
    for f in [gguf_f16, gguf_out]:
        if f.exists():
            logger.info(f"  {f.name}: {f.stat().st_size/1024**2:.0f} MB")
    
    logger.info(f"\n최종 제출용 파일: {gguf_out}")
    logger.info("\nHuggingFace 업로드 명령어:")
    logger.info(f"  hf upload [본인레포] {gguf_out}")

if __name__ == "__main__":
    main()
