"""Competition dataset adapter - AI 코딩 에이전트 의사결정 데이터셋.

태스크: 코딩 세션 컨텍스트를 입력받아 다음 행동(action)을 예측.

입력: session_meta + history + current_prompt
출력: 14개 액션 중 하나
  read_file, write_file, edit_file, apply_patch,
  grep_search, glob_pattern, list_directory,
  run_bash, run_tests, lint_or_typecheck,
  web_search, ask_user, plan_task, respond_only
"""

import json
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters.base_adapter import DatasetAdapter, ToolSchema, Example
from adapters.registry import register_adapter

logger = logging.getLogger(__name__)

ACTIONS = [
    "read_file", "write_file", "edit_file", "apply_patch",
    "grep_search", "glob_pattern", "list_directory",
    "run_bash", "run_tests", "lint_or_typecheck",
    "web_search", "ask_user", "plan_task", "respond_only",
]

SYSTEM_PROMPT = """You are an AI coding assistant decision model.
Given a coding session context, output exactly one action label.

Available actions:
- read_file: Open and read a specific file
- write_file: Create a new file
- edit_file: Modify an existing file
- apply_patch: Apply a multi-file patch
- grep_search: Search for a pattern in code
- glob_pattern: Find files matching a pattern
- list_directory: List directory contents
- run_bash: Execute a shell command
- run_tests: Run test suite
- lint_or_typecheck: Run linter or type checker
- web_search: Search the web for information
- ask_user: Ask the user a clarifying question
- plan_task: Create a step-by-step plan before acting
- respond_only: Reply with text only, no tool use needed

Output only the action label, nothing else."""


@register_adapter("competition")
class CompetitionAdapter(DatasetAdapter):
    """Adapter for AI 의사결정 대회 데이터셋."""

    def __init__(self):
        super().__init__(
            dataset_name="competition_local",
            version="1.0"
        )
        self.raw_dir = Path("./data_alt")
        self.labels: Dict[str, str] = {}

    def download(self, cache_dir: str) -> None:
        """로컬 데이터 파일 로드 (다운로드 불필요)."""
        logger.info("로컬 데이터 파일 로드 중...")

        labels_path = self.raw_dir / "train_labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"train_labels.csv not found at {labels_path}\n"
                "data/raw/ 폴더에 대회 데이터를 복사해주세요."
            )

        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.labels[row["id"]] = row["action"]

        logger.info(f"✓ 레이블 로드: {len(self.labels)}개")

        # 액션 분포 출력
        from collections import Counter
        dist = Counter(self.labels.values())
        logger.info("액션 분포:")
        for action, count in sorted(dist.items(), key=lambda x: -x[1]):
            logger.info(f"  {action:20s}: {count:5d}")

    def load_tools(self) -> List[ToolSchema]:
        """분류 태스크이므로 tool schema 불필요."""
        return []

    def load_examples(self, split: str = "train") -> List[Example]:
        """JSONL 파일에서 예제 로드 후 train/val/test 분할."""
        if split == "test":
            jsonl_path = self.raw_dir / "test.jsonl"
        else:
            jsonl_path = self.raw_dir / "train.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(f"{jsonl_path} not found")

        raw_examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_examples.append(json.loads(line))

        if split == "test":
            examples = [
                Example(
                    prompt=self._format_context(item),
                    tool_call={"action": ""},  # 레이블 없음
                    metadata={"id": item["id"]}
                )
                for item in raw_examples
            ]
        else:
            # train/val 분할: 90/10
            labeled = [
                item for item in raw_examples
                if item["id"] in self.labels
            ]
            split_idx = int(len(labeled) * 0.9)
            if split == "train":
                items = labeled[:split_idx]
            else:  # validation
                items = labeled[split_idx:]

            examples = [
                Example(
                    prompt=self._format_context(item),
                    tool_call={"action": self.labels[item["id"]]},
                    metadata={"id": item["id"]}
                )
                for item in items
            ]

        self.examples[split] = examples
        logger.info(f"✓ {split}: {len(examples)}개 예제 로드")
        return examples

    def _format_context(self, item: Dict[str, Any]) -> str:
        """세션 컨텍스트를 자연어 프롬프트로 변환."""
        meta = item["session_meta"]
        workspace = meta.get("workspace", {})
        lang_mix = workspace.get("language_mix", {})
        top_lang = max(lang_mix, key=lang_mix.get) if lang_mix else "unknown"

        # 세션 메타 요약
        context_parts = [
            f"[Session Context]",
            f"User tier: {meta.get('user_tier', 'unknown')}",
            f"Language preference: {meta.get('language_pref', 'unknown')}",
            f"Primary language: {top_lang} ({lang_mix.get(top_lang, 0)*100:.0f}%)",
            f"Codebase size: {workspace.get('loc', 0):,} lines",
            f"Git dirty: {workspace.get('git_dirty', False)}",
            f"CI status: {workspace.get('last_ci_status', 'none')}",
            f"Open files: {', '.join(workspace.get('open_files', [])) or 'none'}",
            f"Budget tokens remaining: {meta.get('budget_tokens_remaining', 0):,}",
            f"Turn index: {meta.get('turn_index', 0)}",
        ]

        # 대화 히스토리
        history = item.get("history", [])
        if history:
            context_parts.append("\n[Conversation History]")
            for turn in history[-6:]:  # 최근 6턴만 (토큰 절약)
                role = turn.get("role", "")
                if role == "user":
                    content = turn.get("content", "")[:200]
                    context_parts.append(f"User: {content}")
                elif role == "assistant_action":
                    name = turn.get("name", "")
                    summary = turn.get("result_summary", "")[:100]
                    context_parts.append(f"Assistant action: {name} → {summary}")

        # 현재 프롬프트
        context_parts.append(f"\n[Current User Request]")
        context_parts.append(item.get("current_prompt", ""))

        return "\n".join(context_parts)

    def convert_to_chat_format(
        self,
        example: Example,
        tools: Optional[List[ToolSchema]] = None
    ) -> Dict[str, Any]:
        """chat 형식으로 변환 (SFT 학습용)."""
        action = example.tool_call.get("action", "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example.prompt},
            {"role": "assistant", "content": action},
        ]

        return {"messages": messages}

    def convert_to_inference_format(
        self,
        prompt: str,
        tools: Optional[List[ToolSchema]] = None
    ) -> List[Dict[str, str]]:
        """추론용 메시지 포맷."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def parse_prediction(
        self,
        output: str,
        tools: Optional[List[ToolSchema]] = None
    ) -> Dict[str, Any]:
        """모델 출력에서 액션 추출."""
        output = output.strip().lower()
        # 정확히 일치하는 액션 찾기
        for action in ACTIONS:
            if action in output:
                return {"action": action}
        return {"action": output, "warning": "unknown action"}

    def get_statistics(self) -> Dict[str, Any]:
        from collections import Counter
        return {
            "total_examples": len(self.labels),
            "num_actions": len(ACTIONS),
            "actions": ACTIONS,
            "label_distribution": dict(Counter(self.labels.values())),
        }

    def get_split_info(self) -> Dict[str, int]:
        total = len(self.labels)
        return {
            "train": int(total * 0.9),
            "validation": total - int(total * 0.9),
        }
