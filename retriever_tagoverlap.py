from collections import defaultdict
from typing import List
from data_loader import get_all_test
from tag_norm import TagNormalization

class QueryState:
    def __init__(self):
        self.uuid_2_cnt = defaultdict(int)
        
class TagOverlapRetriever:
    def __init__(self, uuid_l:List[str], doc_l:List[List[str]]):
        self.uuid_l = uuid_l
        self.doc_l = doc_l
        self.tag_2_uuid_set = defaultdict(set)
        
        self.uuid_2_doc = dict()
        for uuid, doc in zip(uuid_l, doc_l):
            self.uuid_2_doc[uuid] = doc
            
        for idx, doc in enumerate(doc_l):
            for tag in doc:
                if len(tag) == 0:
                    continue
                self.tag_2_uuid_set[tag].add(uuid_l[idx])

    def _process_entity(self, kw: str, tag: str, uuid_set: set, qs: QueryState):
        if kw in tag or tag in kw:
            for uuid in uuid_set:
                qs.uuid_2_cnt[uuid] += 1

    def _process_normal(self, kw: str, tag: str, uuid_set: set, qs: QueryState):
        if kw == tag:
            for uuid in uuid_set:
                qs.uuid_2_cnt[uuid] += 1

    def _calc_jaccard(self, kw_set: set, doc_w_set: set):
        return len(kw_set & doc_w_set) # / len(kw_set | doc_w_set)
        
    def get_result(self, kw_list: List[str], top_k = 100) -> List[str]:
        qs = QueryState()
        for kw in kw_list:
            for tag, uuid_set in self.tag_2_uuid_set.items():
                if tag[0] >= 'A' and tag[0] <= 'Z':
                    self._process_entity(kw, tag, uuid_set, qs)
                else:
                    self._process_normal(kw, tag, uuid_set, qs)

        _l = []
        kw_set = set(kw_list)
        for uuid, _ in qs.uuid_2_cnt.items():
            doc = self.uuid_2_doc[uuid]
            doc_w_set = set(doc)
            _l.append((uuid, self._calc_jaccard(kw_set, doc_w_set)))

        _l.sort(key=lambda x: x[1], reverse=True)
        ret = [uuid for uuid, _ in _l[:top_k]]
        
        return ret
        
def test():
    all_uuid_l, all_doc_l = get_all_test()
    uuid_2_doc = {}
    for uuid, doc in zip(all_uuid_l, all_doc_l):
        uuid_2_doc[uuid] = doc
        
    retriever = TagOverlapRetriever(all_uuid_l, all_doc_l)

    tn = TagNormalization()

    kw_l = []
    for kw in ['english', 'korean', 'pop', 'dance', 'feel good']:
        kw_l.extend(tn(kw))

    print(kw_l)
    
    res_uuid_l = retriever.get_result(kw_l, top_k=5)
    for res_uuid in res_uuid_l:
        print(uuid_2_doc[res_uuid])

if __name__ == "__main__":
    test()
    