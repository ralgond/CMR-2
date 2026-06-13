import pandas as pd
from tag_norm import TagNormalization
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from data_loader import get_all_test

class BM25Retriever:
    def __init__(self, uuid: List[str], data: List[List[str]]):
        self.docs = data
        self.uuid = uuid
        self.index = BM25Okapi(self.docs)

    def get_similar_track(self, tags: List[str], top_k=5) -> Tuple[List[str], List[List[str]]]:
        '''
        计算相似的track，tags会包含歌曲名，歌手名，album名，tags
        '''
        scores = self.index.get_scores(tags)
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        ret_uuid = [self.uuid[idx] for idx in top_k_indices]
        ret_docs = [self.docs[idx] for idx in top_k_indices]
        
        return ret_uuid, ret_docs
    
def test_calc_similar_track():
    uuid_l, doc_l = get_all_test()

    bm25_retriever = BM25Retriever(uuid_l, doc_l)
    
    uuid_l, doc_l = bm25_retriever.get_similar_track(["Guru's Jazzmatazz"])
    for uuid, doc in zip(uuid_l, doc_l):
        print(uuid, doc)

if __name__ == "__main__":
    # test_calc_similar_track()
    test_calc_similar_track()
    
        