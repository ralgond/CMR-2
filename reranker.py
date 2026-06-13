# Requires transformers>=4.51.0
import torch
from modelscope import AutoModel, AutoTokenizer, AutoModelForCausalLM

MODEL_PATH="/root/.cache/modelscope/hub/models/Qwen/Qwen3-Reranker-0___6B"

class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 1024
        
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
                
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'
        
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def get_scores(self, query, docs):
        return_scores = []
        for doc in docs:
            pairs = [self.format_instruction(self.task, query, doc) ]
            inputs = self.process_inputs(pairs)
            scores = self.compute_logits(inputs)
            return_scores.extend(scores)
        return return_scores

def test():
    queries = ["What is the capital of China?",
                "Explain gravity",
            ]

    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    reranker = Reranker()

    pairs = [reranker.format_instruction(reranker.task, query, doc) for query, doc in zip(queries, documents)]

    # Tokenize the input texts
    inputs = reranker.process_inputs(pairs)
    scores = reranker.compute_logits(inputs)

    print("scores: ", scores)

if __name__ == "__main__":
    test()