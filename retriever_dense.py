from embedder import QwenEmbedder
import numpy as np
import pandas as pd
import faiss
from utils import weighted_rrf
from data_loader import get_all_test
from reranker import Reranker

class DenseRetriever:
    def __init__(self, uuid_l, doc_l):
        self.qwen_embedder = QwenEmbedder()
        # self.reranker = Reranker()
        
        self.uuid_l = uuid_l
        self.doc_l = doc_l

        df = pd.read_parquet('data/Track-Embedding/test_tracks-00000-of-00001.parquet')

        l = [emb for emb in df['attributes-qwen3_embedding_0.6b'] if emb.shape[0] > 0]
        self.attr_emb = np.vstack(l).astype(np.float32)
        faiss.normalize_L2(self.attr_emb)
        print("attr_emb.shape:", self.attr_emb.shape)
        self.attr_index = faiss.IndexFlatL2(1024)
        self.attr_index.add(self.attr_emb)
        self.attr_1024_idx = [idx for idx, emb in enumerate(df['attributes-qwen3_embedding_0.6b']) if emb.shape[0] > 0]

        l = [emb for emb in df['metadata-qwen3_embedding_0.6b'] if emb.shape[0] > 0]
        self.meta_emb = np.vstack(l).astype(np.float32)
        faiss.normalize_L2(self.meta_emb)
        print("meta_emb.shape:", self.meta_emb.shape)
        self.meta_index = faiss.IndexFlatL2(1024)
        self.meta_index.add(self.meta_emb)
        self.meta_1024_idx = [idx for idx, emb in enumerate(df['metadata-qwen3_embedding_0.6b']) if emb.shape[0] > 0]

        l = [emb for emb in df['lyrics-qwen3_embedding_0.6b'] if emb.shape[0] > 0]
        self.lyrics_emb = np.vstack(l).astype(np.float32)
        faiss.normalize_L2(self.lyrics_emb)
        print("lyrics_emb.shape:", self.lyrics_emb.shape)
        self.lyrics_index = faiss.IndexFlatL2(1024)
        self.lyrics_index.add(self.lyrics_emb)
        self.lyrics_1024_idx = [idx for idx, emb in enumerate(df['lyrics-qwen3_embedding_0.6b']) if emb.shape[0] > 0]
    

    def get_similar_track(self, query, top_k=100):
        emb = np.array([self.qwen_embedder(query)])

        faiss.normalize_L2(emb)

        norm_emb = emb

        track_id_l = self.uuid_l

        _attr_scores, _attr_ids = self.attr_index.search(norm_emb, 200)
        attr_scores, attr_ids = _attr_scores[0], _attr_ids[0]
        sorted_idx = np.argsort(attr_scores)
        sorted_attr_ids = [attr_ids[idx] for idx in sorted_idx]
        attr_uuid_l = []
        for id in sorted_attr_ids:
            attr_uuid_l.append(track_id_l[self.attr_1024_idx[id]])
        
        
        _meta_scores, _meta_ids = self.meta_index.search(norm_emb, 200)
        meta_scores, meta_ids = _meta_scores[0], _meta_ids[0]
        sorted_idx = np.argsort(meta_scores)
        sorted_meta_ids = [meta_ids[idx] for idx in sorted_idx]
        meta_uuid_l = []
        for id in sorted_meta_ids:
            meta_uuid_l.append(track_id_l[self.meta_1024_idx[id]])

        
        _lyrics_scores, _lyrics_ids = self.lyrics_index.search(norm_emb, 200)
        lyrics_scores, lyrics_ids = _lyrics_scores[0], _lyrics_ids[0]
        sorted_idx = np.argsort(lyrics_scores)
        sorted_lyrics_ids = [lyrics_ids[idx] for idx in sorted_idx]
        lyrics_uuid_l = []
        for id in sorted_lyrics_ids:
            lyrics_uuid_l.append(track_id_l[self.lyrics_1024_idx[id]])

        res_uuid_l = weighted_rrf([attr_uuid_l, meta_uuid_l, lyrics_uuid_l], [0.1, 0.8, 0.1])

        res_doc_l = []
        for uuid, doc in zip(self.uuid_l, self.doc_l):
            for res_uuid in res_uuid_l:
                if uuid == res_uuid:
                    res_doc_l.append(doc)
                    break
                    
        res_u_l, res_d_l = res_uuid_l[:top_k], res_doc_l[:top_k]
        return res_u_l, res_d_l
        
        # _res_d_l = [' '.join(res_d) for res_d in res_d_l]
        # scores = self.reranker.get_scores(query, _res_d_l)
        # sorted_idx = np.argsort(np.array(scores))[::-1]
        # res_u_l2, res_d_l2 = [], []
        # for idx in sorted_idx:
        #     res_u_l2.append(res_u_l[idx])
        #     res_d_l2.append(res_d_l[idx])
        # return res_u_l2, res_d_l2

def test():
    uuid_l, doc_l = get_all_test()

    de = DenseRetriever(uuid_l, doc_l)

    res_uuid_l, _ = de.get_similar_track("Play me some music that makes me want to dance and feel good.")

    for res_uuid in res_uuid_l[:5]:
        for _u, _d in zip(uuid_l, doc_l):
            if res_uuid == _u:
                print(_d)

if __name__ == "__main__":
    test()
        
