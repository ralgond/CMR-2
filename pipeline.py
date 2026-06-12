import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from intent_parser import IntentParser
from retriever_bm25 import BM25Retriever, get_all_test
from tag_norm import TagNormalization
from filter import Filter

def load_test_user_cf_bpr():
    df1 = pd.read_parquet('data/User-Embedding/test_cold-00000-of-00001.parquet')
    df2 = pd.read_parquet('data/User-Embedding/test_warm-00000-of-00001.parquet')
    test_user_df = pd.concat([df1, df2], ignore_index=True)

    ret = {}
    for idx, row in test_user_df.iterrows():
        ret[row.get('user_id')] = row.get('cf-bpr')

    return ret

user_cf_bpr_d = load_test_user_cf_bpr()

def load_test_track_cf_bpr():
    test_track_df = pd.read_parquet('data/Track-Embedding/test_tracks-00000-of-00001.parquet', columns=['track_id', 'cf-bpr'])

    ret = {}
    for idx, row in test_track_df.iterrows():
        ret[row.get('track_id')] = row.get('cf-bpr')

    return ret

track_cf_bpr_d = load_test_track_cf_bpr()



all_uuid_l, all_doc_l = get_all_test()

all_bm25_retriever = BM25Retriever(all_uuid_l, all_doc_l)

tn = TagNormalization()
ip = IntentParser()
df = pd.read_parquet('data/Challenge-Blind-A/test-00000-of-00001.parquet')
all_output = []
for session_id, user_id, user_profile, session_data in tqdm(
    zip(df['session_id'].tolist(), df['user_id'].tolist(), df['user_profile'].tolist(), df['conversations'].tolist()), total=len(df)):
    # 基于DeepSeek做意图检索
    intent_result = ip(session_data.tolist())
    print(intent_result)
    

    # 组建query
    bm25_query = []
    for artist in intent_result['pos_artists']:
        bm25_query.append(artist)

    norm_tags = []
    for tag in intent_result['pos_tags']:
        _l = tn(tag)
        norm_tags.extend(_l)

    # norm_tags.append(user_profile['age_group'])
    norm_tags.extend(tn(user_profile['preferred_language']))
    norm_tags.extend(tn(user_profile['preferred_musical_culture']))

    bm25_query.extend(norm_tags)

    # 检索
    uuid_l, doc_l = all_bm25_retriever.get_similar_track(bm25_query, top_k=200)
    print(f"[BM25] retrieved {len(uuid_l)}.")

    # 组建filter
    filter = Filter(uuid_l, doc_l)
    uuid_l2, doc_l2 = filter.filter_in_match(intent_result['neg_artists'])
    print(f"[Filter.filter_in_match] left {len(uuid_l2)}.")

    filter = Filter(uuid_l2, doc_l2)
    uuid_l3, doc_l3 = filter.filter_exact_match(intent_result['neg_tags'])
    print(f"[Filter.filter_exact_match] left {len(uuid_l3)}.")
    
    # 根据歌手进行去重，歌手的名字在doc[1]
    uuid_l4, doc_l4 = [], []
    seen_artist_d = dict()
    for uuid, doc in zip(uuid_l3, doc_l3):
        artist = doc[1]
        if artist not in seen_artist_d or seen_artist_d[artist] < 5:
            uuid_l4.append(uuid)
            doc_l4.append(doc)
            seen_artist_d[artist] = seen_artist_d.get(artist, 0) + 1

    print(f"[Drop dup artist] left {len(uuid_l4)}.")

    # 利用bpr最后的精排
    # uuid_l5 = []
    # doc_l5 = []
    # user_cf_bpr = user_cf_bpr_d.get(user_id, np.array([]))
    # cf_scores = []
    # if user_cf_bpr.shape[0] == 128:
    #     for uuid, doc in zip(uuid_l4, doc_l4):
    #         track_cf_bpr = track_cf_bpr_d.get(uuid)
    #         if track_cf_bpr.shape[0] != 128:
    #             cf_scores.append(-np.inf)
    #         else:
    #             cf_scores.append(np.dot(user_cf_bpr, track_cf_bpr))
    #     ranked_indices = np.argsort(np.array(cf_scores))[::-1]
    #     for idx in ranked_indices:
    #         uuid_l5.append(uuid_l4[idx])
    #         doc_l5.append(doc_l4[idx])
    # else:
    #     uuid_l5 = uuid_l4.copy()
    #     doc_l5 = [doc.copy() for doc in doc_l4]
    

    turn_l = session_data.tolist()
    turn_number = turn_l[-1]['turn_number']

    output = {}
    output['session_id'] = session_id
    output['user_id'] = user_id
    output['turn_number'] = turn_number
    output['predicted_track_ids'] = uuid_l4[:20]
    output['predicted_response'] = ''
    all_output.append(output)

    # debug
    first_track = output['predicted_track_ids'][0]
    for idx, track in enumerate(all_uuid_l):
        if first_track == track:
            break
    print(all_doc_l[idx])
    print(user_profile['preferred_language'])
    print(user_profile['preferred_musical_culture'])
    
    print("="*80+"\n")

with open("prediction.json", "w+") as of:
    json.dump(all_output, of)