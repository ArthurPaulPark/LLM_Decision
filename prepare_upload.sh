#!/bin/bash
# 제출용 파일 복사 및 업로드 준비 스크립트

echo "[INFO] 최종 제출용 모델 파일을 준비합니다..."
mkdir -p outputs/final_submission

# 가장 성능이 좋았던 V2 모델의 양자화 버전(GGUF)을 업로드 폴더로 복사
cp outputs/qwen15_v2/quantized/model_q3km.gguf outputs/final_submission/

echo "=================================================="
echo "✓ 모델 복사 완료: outputs/final_submission/model_q3km.gguf"
echo "=================================================="
echo "HuggingFace 업로드를 위해 다음 명령어를 실행해 주세요:"
echo ""
echo "python3 upload_to_hub.py \\"
echo "    --repo [본인의_HF_아이디/레포지토리_이름] \\"
echo "    --adapter outputs/final_submission \\"
echo "    --token [본인의_HF_토큰]"
echo ""
echo "예시: python3 upload_to_hub.py --repo guest03/qwen15-1.5b-v2 --adapter outputs/final_submission --token hf_abc123"
