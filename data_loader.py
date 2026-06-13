import pandas as pd
from tag_norm import TagNormalization

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