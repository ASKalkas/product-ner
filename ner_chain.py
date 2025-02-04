import numpy as np
from langchain.chains.base import Chain
from ner import ProductNER

class NERChain(Chain):
    @property
    def input_keys(self):
        return ["tokens"]
    
    @property
    def output_keys(self):
        return ["entities"]
    
    def __init__(self, ner_model, max_seq_length=50):
        self.ner_model = ner_model
        self.max_seq_length = max_seq_length

    def _call(self, inputs):
        tokens = inputs["tokens"]
        entities = self.ner_model.tag(tokens[0])
        
        return {"entities": entities}