import hashlib
from typing import Dict

class ExactDeduplicator:
    def __init__(self) -> None:
        self.seen_hashes:Dict[str,str] = {}
        self.duplicate_ids: list[tuple[str,str]] = []


    def compute_sha256(self,text:str)->str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def is_duplicate(self,doc_id:str,content:str)->bool:
        content_hash = self.compute_sha256(content)

        if content_hash in self.seen_hashes:
            self.duplicate_ids.append((doc_id,self.seen_hashes[content_hash]))
            return True

        self.seen_hashes[content_hash] = doc_id
        return False