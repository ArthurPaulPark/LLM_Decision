"""Dataset adapters for universal framework."""

from adapters.base_adapter import DatasetAdapter
from adapters.xlam_adapter import XLAMAdapter
from adapters.competition_adapter import CompetitionAdapter
from adapters.registry import get_adapter, register_adapter

__all__ = ["DatasetAdapter", "XLAMAdapter", "CompetitionAdapter", "get_adapter", "register_adapter"]

