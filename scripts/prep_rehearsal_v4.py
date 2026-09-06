import json
import random
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Create V4 Rehearsal dataset (Hard + Easy mixing)")
    parser.add_argument("--original_data", type=str, default="data/processed/train.json", help="Original full train data")
    parser.add_argument("--hard_data", type=str, default="data/processed_hard/train.json", help="Hard samples data")
    parser.add_argument("--output_dir", type=str, default="data/processed_v4_rehearsal", help="Output directory")
    parser.add_argument("--easy_samples", type=int, default=18000, help="Number of easy samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    
    # 1. 파일 로드
    logger.info(f"Loading original data from {args.original_data}...")
    with open(args.original_data, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    logger.info(f"Loading hard data from {args.hard_data}...")
    with open(args.hard_data, 'r', encoding='utf-8') as f:
        hard_data = json.load(f)
        
    logger.info(f"Original samples: {len(original_data)}, Hard samples: {len(hard_data)}")

    # 2. Hard sample ID 세트 생성 (중복 방지)
    # 딕셔너리로 직렬화/비교 시 오차가 생길 수 있으므로, instruction 또는 text 전체를 해시나 키로 쓰거나
    # 데이터 구조에 ID가 없다면 텍스트 기반 중복 제거를 수행.
    # Qwen 포맷 구조: [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]
    
    def get_text_key(item):
        # user의 content를 고유 키로 사용
        for msg in item.get("messages", []):
            if msg["role"] == "user":
                return msg["content"]
        return str(item)
    
    hard_keys = {get_text_key(item) for item in hard_data}
    
    # 3. Easy sample 추출 (Hard sample을 제외한 나머지 풀에서 추출)
    easy_pool = [item for item in original_data if get_text_key(item) not in hard_keys]
    logger.info(f"Remaining easy samples in pool: {len(easy_pool)}")
    
    if len(easy_pool) < args.easy_samples:
        logger.warning(f"Easy pool ({len(easy_pool)}) is smaller than requested samples ({args.easy_samples}). Using all available easy samples.")
        selected_easy = easy_pool
    else:
        selected_easy = random.sample(easy_pool, args.easy_samples)
        
    # 4. 병합 및 셔플
    rehearsal_data = hard_data + selected_easy
    random.shuffle(rehearsal_data)
    
    # 5. 저장
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "train.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(rehearsal_data, f, ensure_ascii=False, indent=2)
        
    logger.info("="*50)
    logger.info("Rehearsal Dataset Creation Summary")
    logger.info("="*50)
    logger.info(f"Hard samples  : {len(hard_data)} (approx. {len(hard_data)/len(rehearsal_data)*100:.1f}%)")
    logger.info(f"Easy samples  : {len(selected_easy)} (approx. {len(selected_easy)/len(rehearsal_data)*100:.1f}%)")
    logger.info(f"Total samples : {len(rehearsal_data)}")
    logger.info(f"Output saved  : {output_path}")

if __name__ == "__main__":
    main()
