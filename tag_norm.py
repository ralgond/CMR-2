from typing import List
import re
import Stemmer

class TagNormalization():
    def __init__(self):
        self._number_single_quote_s_map = {'2000s':'00s', '2010s':'10s', '1990s':'90s', '1980s':'80s', '1970s':'70s'}
        self.stemmer = Stemmer.Stemmer("english")

    def __call__(self, tag) -> List[str]:
        def _number_single_quote_s_2_number_s(tag):
            if re.match(r'^[0-9]{2,4}\'s$', tag):
                return tag.replace('\'', '')
            return tag

        def _number_s_shorter(tag):
            if tag in self._number_single_quote_s_map:
                return self._number_single_quote_s_map[tag]
            return tag

        def _stem(tag):
            tag = tag.strip()
            _tags = tag.split()
            if len(_tags) == 1:
                return self.stemmer.stemWords([tag])[0]
            else:
                return " ".join(self.stemmer.stemWords(_tags))
        
        if len(tag) == 0:
            return []
        tags1 = tag.split('/')
        tags2 = list(map(lambda x: x.lower(), tags1))
        tags3 = list(map(_number_single_quote_s_2_number_s, tags2))
        tags4 = list(map(_number_s_shorter, tags3))
        tags5 = list(map(lambda x: x.replace('-', ' ').strip(), tags4))
        tags6 = list(map(_stem, tags5))
        return tags6


def test():
    tn = TagNormalization()
    assert ["abc"] == tn("abc"), "abc"
    assert ["ab", "c"] == tn("ab/c"), "ab/c"
    assert ["abc"] == tn("Abc"), "Abc"
    assert ["80s"] == tn("80's"), "80's"
    assert ["00s"] == tn("2000's"), "2000's"
    assert ["10s"] == tn("2010s"), "2010s"
    assert ["hip pop"] == tn("hip-pop"), "hip-pop"
    assert ["danc"] == tn("dancing"), "dancing"
    assert ['male vocalist'] == tn("male vocalists"),  "male vocalists"

if __name__ == "__main__":
    test()
        
        

    