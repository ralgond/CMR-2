import pandas as pd
import json
import pprint
from collections import defaultdict

# 读取指定的单列数据
def debug_train():
    df = pd.read_parquet('data/Challenge-Data/train-00000-of-00001.parquet')
    
    data = df['conversations'].tolist()[0]
    pprint.pprint(data.tolist())
    
    #print(json.loads(data))

def debug_track_emb():
    df = pd.read_parquet('data/Track-Embedding/test_tracks-00000-of-00001.parquet')
    print(df.columns)

def debug_track_meta():
    d = defaultdict(int)
    df = pd.read_parquet('data/Track-Metadata/test_tracks-00000-of-00001.parquet')
    print(df.columns)
    data = df['tag_list'].tolist()
    for tag_list in df['tag_list']:
        for tag in tag_list:
            d[tag] += 1

    l = sorted([(tag,cnt) for tag, cnt in d.items()], key=lambda x:x[1], reverse=True)

    for tag,cnt in l:
        print(tag,"\t",cnt)

def debug_track_meta_print_70s():
    d = defaultdict(int)
    df = pd.read_parquet('data/Track-Metadata/test_tracks-00000-of-00001.parquet')
    print(df.columns)
    tag_list_l = df['tag_list'].tolist()
    track_name_l = df['track_name'].tolist()

    for track_name, tag_list in zip(track_name_l, tag_list_l):
        for tag in tag_list:
            if tag == '70s':
                print(track_name, tag_list)

def debug_tag_normalization():
    from tag_norm import TagNormalization
    tn = TagNormalization()
    d = defaultdict(int)
    df = pd.read_parquet('data/Track-Metadata/test_tracks-00000-of-00001.parquet')
    print(df.columns)
    data = df['tag_list'].tolist()
    for tag_list in df['tag_list']:
        for tag in tag_list:
            _tags = tn(tag)
            for _tag in _tags:
                d[_tag] += 1

    l = sorted([(tag,cnt) for tag, cnt in d.items()], key=lambda x:x[1], reverse=True)

    for tag,cnt in l:
        print(tag,"\t",cnt)

debug_tag_normalization()
