from pydantic import BaseModel
from typing import Optional, List

class FixRequest(BaseModel):
    file_path: str
    action: str # 'delete', 'move', 'ignore'
    new_label: Optional[str] = None

class BatchFixRequest(BaseModel):
    file_paths: list[str]
    action: str
    new_label: Optional[str] = None

class BatchItem(BaseModel):
    file_path: str
    new_label: str

class BatchSuggestionRequest(BaseModel):
    items: List[BatchItem]
