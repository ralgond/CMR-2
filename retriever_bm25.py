import pandas as pd
from tag_norm import TagNormalization
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
from typing import List, Tuple

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


def get_all_test():
    data_path='data/Track-Metadata/test_tracks-00000-of-00001.parquet'
    df = pd.read_parquet(data_path)

    tn = TagNormalization()
    uuid_l = []
    doc_l = []
    for idx, track in df.iterrows():
        uuid_l.append(track['track_id'])
        track_name = track['track_name'].tolist()
        artist_name = track['artist_name'].tolist()
        album_name = track['album_name'].tolist()
        tags = track['tag_list'].tolist()
        tags_norm = []
        for tag in tags:
            tmp_tags = tn(tag)
            tags_norm.extend(tmp_tags)
            # 切开phrase
            # for _tag in tmp_tags:
            #     if ' ' in _tag:
            #         norm_tags.extend(_tag.split())
    
        doc = track_name + artist_name + album_name + tags_norm
        doc_l.append(doc)
    
    return uuid_l, doc_l
    
def test_calc_similar_track():
    uuid_l, doc_l = get_all_test()

    bm25_retriever = BM25Retriever(uuid_l, doc_l)
    
    uuid_l, doc_l = bm25_retriever.get_similar_track(["Guru's Jazzmatazz"])
    for uuid, doc in zip(uuid_l, doc_l):
        print(uuid, doc)

if __name__ == "__main__":
    # test_calc_similar_track()
    test_calc_similar_track()
    
        