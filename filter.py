from rapidfuzz import fuzz

class Filter:
    def __init__(self, uuid_l, doc_l):
        self.uuid_l = uuid_l
        self.doc_l = doc_l

    def filter_in_match(self, keywords):
        filter_idx = set()
        for kw in keywords:
            for idx, _doc in enumerate(self.doc_l):
                for term in _doc:
                    if kw in term:
                        filter_idx.add(idx)
                        break

        ret_uuid_l = []
        ret_doc_l = []
        for idx, _doc in enumerate(self.doc_l):
            if idx in filter_idx:
                continue
            ret_uuid_l.append(self.uuid_l[idx])
            ret_doc_l.append(self.doc_l[idx])
            
        return ret_uuid_l, ret_doc_l

    def filter_in_match2(self, keywords):
        filter_idx = set()
        for kw in keywords:
            for idx, _doc in enumerate(self.doc_l):
                for term in _doc:
                    if kw in term or term in kw:
                        filter_idx.add(idx)
                        break

        ret_uuid_l = []
        ret_doc_l = []
        for idx, _doc in enumerate(self.doc_l):
            if idx in filter_idx:
                continue
            ret_uuid_l.append(self.uuid_l[idx])
            ret_doc_l.append(self.doc_l[idx])
            
        return ret_uuid_l, ret_doc_l

    def filter_exact_match(self, keywords):
        filter_idx = set()
        for kw in keywords:
            for idx, _doc in enumerate(self.doc_l):
                for term in _doc:
                    if kw == term:
                        filter_idx.add(idx)
                        break
        
        ret_uuid_l = []
        ret_doc_l = []
        for idx, _doc in enumerate(self.doc_l):
            if idx in filter_idx:
                continue
            ret_uuid_l.append(self.uuid_l[idx])
            ret_doc_l.append(self.doc_l[idx])
            
        return ret_uuid_l, ret_doc_l

    def filter_fuzz_match(self, keywords):
        filter_idx = set()
        for kw in keywords:
            for idx, _doc in enumerate(self.doc_l):
                for term in _doc:
                    if fuzz.ratio(kw, term) >= 90:
                        filter_idx.add(idx)
                        break
        
        ret_uuid_l = []
        ret_doc_l = []
        for idx, _doc in enumerate(self.doc_l):
            if idx in filter_idx:
                continue
            ret_uuid_l.append(self.uuid_l[idx])
            ret_doc_l.append(self.doc_l[idx])
            
        return ret_uuid_l, ret_doc_l 
                

def test():
    uuid_l = ['A', 'B', 'C']
    doc_l = [
        ['The Sun', 'yellow', '80s'],
        ['The Moon', 'white', '90s'],
        ['The Star', 'blink', '2000s']
    ]

    filter = Filter(uuid_l, doc_l)

    uuid, doc = filter.filter_in_match(['Sun', 'The Star', 'Wall'])
    assert uuid == ['B']
    assert doc == [['The Moon', 'white', '90s']]

    uuid, doc = filter.filter_exact_match(['Sun', 'The Star', 'Wall'])
    assert uuid == ['A', 'B']
    assert doc == [['The Sun', 'yellow', '80s'], ['The Moon', 'white', '90s']]

    uuid, doc = filter.filter_exact_match(['yellow'])
    assert uuid == ['B', 'C']
    assert doc == [['The Moon', 'white', '90s'], ['The Star', 'blink', '2000s']]

    uuid, doc = filter.filter_exact_match(['90s'])
    assert uuid == ['A', 'C']
    assert doc == [['The Sun', 'yellow', '80s'], ['The Star', 'blink', '2000s']]


if __name__ == "__main__":
    test()