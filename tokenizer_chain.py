from langchain.chains.base import Chain
from tokenizer import Tokenizer

class TokenizerChain(Chain):
    @property
    def input_keys(self):
        return ["text"]
    
    @property
    def output_keys(self):
        return ["tokens"]
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def _call(self, inputs):
        text = inputs["text"]
        tokens = self.tokenizer.tokenize([text])
        return {"tokens": tokens}
    
    def run(self, text):
        return self._call({"text": text})
