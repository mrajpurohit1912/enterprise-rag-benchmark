import re
import unicodedata

def normalize_text(text:str) -> str:
    if not text or not isinstance(text,str):
        return ""

    text = unicodedata.normalize("NFKC",text)

    text = text.replace("\r\n","\n").replace("\r","\n")

    text = re.sub(r"[\u00a0\u2007\xa0]", " ", text)                                                                                                                                   
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)         

                                                                     
    text = re.sub(r"(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)", r"\1\2", text)            
    
    text = "".join(                                                                                                                                                                   
            ch for ch in text                                                                                                                                                             
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")                                                                                                                   
        )                                                                                                                                                                                 
                                                                                                                                                                                          
    text = re.sub(r"[ \t]+", " ", text)                                                                                                                                               
                                                                                                                                                                                          
    text = re.sub(r"\n{3,}", "\n\n", text)                                                                                                                                            
                                                                                                                                                                                          
    text = "\n".join(line.strip() for line in text.split("\n"))                                                                                                                       
                                                                                                                                                                                          
    return text.strip()                             