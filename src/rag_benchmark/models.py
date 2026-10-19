from pydantic import BaseModel


class RawQuestions(BaseModel):
    question_id:str
    question_type:str
    source_types:str
    question:str
    expected_doc_ids:str
    gold_answer:str
    answer_facts:str

class RawDocuments(BaseModel):                                                                                                                                                  
    doc_id: str                                                                                                                                                                
    source_type: str                                                                                                                                                           
    title: str                                                                                                                                                                 
    content: str   

class CleanedQuestions(RawQuestions):
    raw_questions:RawQuestions
    normalized_questions:str

class CleanedDocuments(RawDocuments):
    raw_documents:RawDocuments
    cleaned_title:str
    cleaned_content:str

class ParentChunk(BaseModel):
    parent_id: str
    doc_id: str
    source_type: str
    title: str
    parent_text: str

class ChildChunk(BaseModel):
    child_id: str
    parent_id: str
    doc_id: str
    source_type: str
    title: str
    child_text: str
    dense_vector: list[float] | None = None