"""
retriever.py
三阶段检索：
  Stage 1a: Qwen3 cosine ANN 召回 (语义路)
  Stage 1b: BM25 召回 (关键词精确匹配路)
  Stage 2 : 多信号加权重排序，取 top-k

BM25 索引字段：
  artist_name + track_name + album_name + tag_list
BM25 查询词：
  artists + tags + mood_keywords + era（从 query_dict 提取）
"""

import json
import os
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from openai import OpenAI
from rank_bm25 import BM25Okapi

from embedder import encode_queries, encode_cover_query

# 触发封面召回路的关键词
COVER_KEYWORDS = {
    "cover", "album cover", "album art", "artwork", "sleeve",
    "jacket", "booklet", "visual", "cover art", "album artwork",
    "封面", "专辑封面", "cover image", "album image",
}

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key"),
    base_url="https://api.deepseek.com",
)
DEEPSEEK_MODEL = "deepseek-chat"


# ──────────────────────────────────────────────
# 1. DeepSeek 结构化查询提取
# ──────────────────────────────────────────────

QUERY_EXTRACTION_SYSTEM = """
You are a music search query extractor for a vector retrieval system.

Given conversation turns (user + music), extract a compact search JSON:

{
  "tags": ["genre/mood/style tags"],
  "artists": ["artist names mentioned positively"],
  "excluded_artists": ["artist names explicitly rejected by user"],
  "mood_keywords": ["energetic", "melancholic", ...],
  "era": "optional decade e.g. 90s",
  "language": "optional e.g. Korean, English",
  "functional_context": "optional e.g. workout, focus, party",
  "confirmed_track_ids": ["UUIDs where next user turn was clearly positive"],
  "rejected_track_ids": ["UUIDs where next user turn was clearly negative"],
  "free_query": "one sentence describing the ideal track to find right now"
}

Rules:
- Track UUIDs appear as the entire content of music turns (role=music)
- Only mark confirmed if the immediately following user turn contains positive words
- Only mark rejected if the immediately following user turn contains negative words
- free_query must be self-contained (merge all session context into it)
- excluded_artists: include any artist the user said they've "heard enough of" or "not this"
- Output valid JSON only, no explanation
"""


def extract_query(conversations: list[dict], up_to_turn: int) -> dict:
    relevant = [t for t in conversations if t["turn_number"] <= up_to_turn]
    conv_text = "\n".join(
        f"[{t['role']}]: {t['content'][:400]}" for t in relevant
    )
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": QUERY_EXTRACTION_SYSTEM},
            {"role": "user", "content": conv_text},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"tags": [], "artists": [], "free_query": conv_text[-300:],
                "confirmed_track_ids": [], "rejected_track_ids": [],
                "excluded_artists": []}


# ──────────────────────────────────────────────
# 2. BM25 索引（懒加载，只建一次）
# ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """简单小写+按非字母数字切分，适合英文音乐元数据。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def _track_to_doc(meta: pd.Series) -> list[str]:
    """
    将一条曲目元数据拼成 token 列表用于 BM25 索引。
    字段权重通过重复体现：artist_name 重复 3 次（最重要），
    track_name 重复 2 次，tag_list 正常权重。
    """
    tokens = []
    # artist_name × 3（用户说"我要 Amon Amarth"时能精确命中）
    artist = str(meta.get("artist_name", ""))
    tokens += _tokenize(artist) * 3

    # track_name × 2
    track = str(meta.get("track_name", ""))
    tokens += _tokenize(track) * 2

    # album_name × 1
    album = str(meta.get("album_name", ""))
    tokens += _tokenize(album)

    # tag_list × 1（genre/mood 标签）
    raw_tags = meta.get("tag_list", [])
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            tokens += _tokenize(str(tag))

    return tokens if tokens else ["unknown"]


class BM25Index:
    """
    懒加载 BM25 索引，首次调用时构建，之后复用。
    """
    def __init__(self):
        self._bm25: BM25Okapi | None = None
        self._index_track_ids: list[str] = []

    def build(self, track_meta: pd.DataFrame) -> None:
        """根据 track_meta 构建索引（仅调用一次）。"""
        print("[BM25] Building index...")
        self._index_track_ids = list(track_meta.index)
        corpus = [_track_to_doc(track_meta.loc[tid]) for tid in self._index_track_ids]
        self._bm25 = BM25Okapi(corpus)
        print(f"[BM25] Index built: {len(self._index_track_ids)} tracks")

    def search(self, query_tokens: list[str], top_k: int) -> list[tuple[str, float]]:
        """返回 [(track_id, bm25_score), ...] 按分数降序。"""
        if self._bm25 is None:
            raise RuntimeError("BM25Index not built yet. Call .build() first.")
        scores = self._bm25.get_scores(query_tokens)          # (N,)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self._index_track_ids[i], float(scores[i]))
                for i in top_indices if scores[i] > 0]


# 全局单例，在 data_loader 之后、第一次 retrieve 之前调用 bm25_index.build()
bm25_index = BM25Index()


def build_bm25_index(track_meta: pd.DataFrame) -> None:
    """外部调用入口（在 predict.py / validate.py 的初始化阶段调用一次）。"""
    bm25_index.build(track_meta)


def _build_bm25_query(query_dict: dict) -> list[str]:
    """
    构建 BM25 查询词列表。
    只包含正向词：artists + tags + mood_keywords + era。
    excluded_artists / rejected_track_ids 不加入，
    否定过滤统一在召回后通过 metadata 完成。
    """
    tokens = []
    for artist in query_dict.get("artists", []):
        tokens += _tokenize(artist) * 2   # 艺术家名权重翻倍
    for tag in query_dict.get("tags", []):
        tokens += _tokenize(tag)
    for mood in query_dict.get("mood_keywords", []):
        tokens += _tokenize(mood)
    era = query_dict.get("era", "")
    if era:
        tokens += _tokenize(era)
    return tokens if tokens else ["music"]


# ──────────────────────────────────────────────
# 3. 查询向量构建（cosine 路）
# ──────────────────────────────────────────────

def _build_free_text(query_dict: dict) -> str:
    """
    构建 dense 查询文本。
    只包含正向信息，否定实体（excluded_artists / rejected_track_ids）
    不进入此文本，避免向量空间被否定方向拉偏。
    """
    parts = []
    free = query_dict.get("free_query", "")
    if free:
        parts.append(free)
    # 只取正向 tags / moods
    tags = query_dict.get("tags", []) + query_dict.get("mood_keywords", [])
    if tags:
        parts.append("Music style and mood: " + ", ".join(tags))
    # 只取正向 artists（excluded_artists 不在此列）
    artists = query_dict.get("artists", [])
    if artists:
        parts.append("Similar to artists: " + ", ".join(artists))
    era = query_dict.get("era", "")
    if era:
        parts.append(f"Era: {era}")
    lang = query_dict.get("language", "")
    if lang:
        parts.append(f"Language: {lang}")
    ctx = query_dict.get("functional_context", "")
    if ctx:
        parts.append(f"Use context: {ctx}")
    # 注意：excluded_artists / rejected_track_ids 不加入此处
    return ". ".join(parts) if parts else "music recommendation"


def build_query_vector(
    query_dict: dict,
    track_ids: list[str],
    track_matrix: np.ndarray,
    confirmed_ids: list[str],
    alpha: float = 0.6,
) -> np.ndarray:
    free_text = _build_free_text(query_dict)
    text_vec  = encode_queries([free_text])[0]

    conf_indices = [track_ids.index(tid) for tid in confirmed_ids if tid in track_ids]
    if conf_indices:
        conf_vecs    = track_matrix[conf_indices]
        conf_centroid = conf_vecs.mean(axis=0)
        norm          = np.linalg.norm(conf_centroid)
        conf_centroid = conf_centroid / norm if norm > 0 else conf_centroid
        q_vec = alpha * text_vec + (1 - alpha) * conf_centroid
    else:
        q_vec = text_vec

    norm = np.linalg.norm(q_vec)
    return q_vec / norm if norm > 0 else q_vec


# ──────────────────────────────────────────────
# 4. 重排序信号
# ──────────────────────────────────────────────

def _tag_overlap_score(query_dict: dict, meta: pd.Series) -> float:
    """
    正向 tag/mood/artist 字面重叠得分，仅用于重排序加分。
    否定过滤已在 _post_recall_filter 中完成，此处不再做排除判断。
    """
    tags    = set(t.lower() for t in query_dict.get("tags", []))
    moods   = set(m.lower() for m in query_dict.get("mood_keywords", []))
    artists = set(a.lower() for a in query_dict.get("artists", []))

    artist_name = str(meta.get("artist_name", "")).lower()

    score = 0.0
    if any(a in artist_name for a in artists):
        score += 10.0

    raw_tags   = meta.get("tag_list", [])
    track_tags = set(t.lower() for t in raw_tags) if isinstance(raw_tags, list) else set()
    score += len(tags  & track_tags) * 2.0
    score += len(moods & track_tags) * 1.5

    era = query_dict.get("era", "")
    if era:
        if era.replace("s", "") in str(meta.get("release_date", "")):
            score += 2.0

    return score


from rapidfuzz import fuzz

# fuzzy 매칭 임계값
FUZZY_ARTIST_THRESHOLD = 85


def _fuzzy_artist_match(artist_name: str, excluded_artists: set[str]) -> bool:
    """
    rapidfuzz 기반 아티스트 fuzzy 매칭.

    전략: token_set_ratio + ratio 이중 체크
    - token_set_ratio >= 85: 토큰 집합 기반 매칭
      ("The Wood Brothers" vs "Wood Brothers", "Amon Amarth (Official)" 처리)
    - ratio >= 40 (일반) / >= 60 (한쪽이 단어 1개, 다른쪽 복수 단어):
      "Coldplay" vs "Cold", "Lil Wayne" vs "Lil" 오탐 방지
    """
    ar = artist_name.lower().strip()
    ar_tokens = ar.split()
    for ex in excluded_artists:
        ex_tokens = ex.split()
        ts = fuzz.token_set_ratio(ex, ar)
        r  = fuzz.ratio(ex, ar)
        min_len = min(len(ex_tokens), len(ar_tokens))
        max_len = max(len(ex_tokens), len(ar_tokens))
        ratio_threshold = 60 if min_len == 1 and max_len > 1 else 40
        if ts >= FUZZY_ARTIST_THRESHOLD and r >= ratio_threshold:
            return True
    return False


def _post_recall_filter(
    candidate_indices: set[int],
    track_ids: list[str],
    track_meta: pd.DataFrame,
    rejected_track_ids: set[str],
    excluded_artists: set[str],
    excluded_track_ids: set[str],
) -> list[int]:
    """
    召回后统一否定过滤，artist 匹配使用 fuzzy（rapidfuzz.partial_ratio）。

    过滤规则：
      1. track_id 在 rejected_track_ids 中（精确，UUID 匹配）
      2. track_id 在 excluded_track_ids 中（精确）
      3. artist_name fuzzy 命中 excluded_artists（threshold=80）
         能处理：大小写、"The X" vs "X"、拼写细微差异
    """
    if not excluded_artists:
        # 无否定艺术家时跳过 fuzzy 逻辑，节省时间
        return [
            idx for idx in candidate_indices
            if track_ids[idx] not in rejected_track_ids
            and track_ids[idx] not in excluded_track_ids
        ]

    passed = []
    for idx in candidate_indices:
        tid = track_ids[idx]

        if tid in rejected_track_ids or tid in excluded_track_ids:
            continue

        if tid in track_meta.index:
            artist_name = str(track_meta.loc[tid].get("artist_name", ""))
            if _fuzzy_artist_match(artist_name, excluded_artists):
                continue

        passed.append(idx)
    return passed


def _popularity_score(meta: pd.Series) -> float:
    """流行度，归一化到 [0, 1]（原始值 0-100）。"""
    pop = meta.get("popularity", 0) or 0
    return float(pop) / 100.0


def _confirmed_similarity_score(
    track_idx: int,
    track_matrix: np.ndarray,
    conf_indices: list[int],
) -> float:
    if not conf_indices:
        return 0.0
    conf_vecs = track_matrix[conf_indices]
    return float((conf_vecs @ track_matrix[track_idx]).mean())


# ──────────────────────────────────────────────
# 5. 主检索函数
# ──────────────────────────────────────────────

def retrieve(
    query_dict: dict,
    track_indices: dict,
    track_meta: pd.DataFrame,
    user_history: list[str],
    session_state=None,          # SessionState 对象，提供动态权重和否定实体
    top_k: int = 20,
    recall_k: int = 300,
    alpha: float = 0.6,
    # 静态默认权重（session_state 存在时被动态覆盖）
    w_meta: float      = 0.20,
    w_attr: float      = 0.15,
    w_lyrics: float    = 0.10,
    w_cover: float     = 0.05,   # 无封面意图时保持低权重，激活时拉高至 0.25
    w_bm25: float      = 0.25,
    w_tag: float       = 0.15,
    w_confirmed: float = 0.07,
    w_pop: float       = 0.03,
) -> list[str]:
    """
    五阶段检索：
      Stage 1a : metadata cosine ANN 召回
      Stage 1b : attributes cosine ANN 召回
      Stage 1c : lyrics cosine ANN 召回
      Stage 1d : cover cosine ANN 召回（SigLIP2，dim=768，检测到封面意图时激活）
      Stage 1e : BM25 召回
      合并去重
      Stage 1f : 召回后统一否定过滤（fuzzy 匹配）
      Stage 2  : 八信号归一化加权重排序 → top_k

    session_state 不为 None 时：
      - meta_w/attr_w/lyric_w 动态覆盖三路向量权重（占总权重 70%）
      - rejected_artists/tracks/tags 合并进过滤集合
      - query_focus 覆盖 free_query
      - specificity_level 微调 meta/attr/lyrics 权重
    """
    # ── 从 session_state 提取动态信息 ──
    if session_state is not None:
        ss = session_state

        # 三路向量权重占总排序权重的 70%，剩余 30% 固定分给 bm25/tag/conf/pop
        VEC_TOTAL = 0.70
        w_meta    = ss.meta_w  * VEC_TOTAL
        w_attr    = ss.attr_w  * VEC_TOTAL
        w_lyrics  = ss.lyric_w * VEC_TOTAL
        w_bm25    = 0.30 * (25 / 50)
        w_tag     = 0.30 * (15 / 50)
        w_confirmed = 0.30 * (7 / 50)
        w_pop     = 0.30 * (3 / 50)

        # ── 三种 intent 模式权重调整 ──
        #
        # Case 1: 用户指定歌手/歌曲 (specificity=high)
        #   → meta（语义匹配歌手名）和 bm25（关键词精确命中）同步拉高
        #     attr/lyrics 相应压低
        #
        # Case 2: 用户要找适合某场景的音乐 (specificity=low, functional_context 存在)
        #   → meta（语义理解场景）和 attr（音乐属性匹配场景）同步拉高
        #     bm25 压低（场景词在 tag 里不一定有对应关键词）
        #
        # Case 3: 用户 query 包含歌词关键字 (lyric_themes 非空)
        #   → lyrics 拉高，其余按比例压缩

        has_artist_intent = (ss.specificity_level == "high")
        has_scene_intent  = (bool(query_dict.get("functional_context"))
                             or ss.specificity_level == "low")
        has_lyric_intent  = bool(query_dict.get("lyric_themes") or ss.lyric_w > 0.35)

        if has_lyric_intent:
            # Case 3: lyrics 拉高到 0.45，其余按比例压缩
            w_lyrics  = 0.45
            remaining = 1.0 - w_lyrics - w_tag - w_confirmed - w_pop
            # meta : attr : bm25 = 3 : 2 : 2（lyrics 优先时 meta 仍略高）
            w_meta  = remaining * (3 / 7)
            w_attr  = remaining * (2 / 7)
            w_bm25  = remaining * (2 / 7)

        elif has_artist_intent:
            # Case 1: meta + bm25 同步拉高，attr/lyrics 压低
            w_meta   = 0.35
            w_bm25   = 0.28
            w_attr   = 0.10
            w_lyrics = 0.05
            # 剩余给 tag/conf/pop（保持原比例）
            w_tag       = 0.12
            w_confirmed = 0.07
            w_pop       = 0.03

        elif has_scene_intent:
            # Case 2: meta + attr 拉高，bm25 控制在 0.2 以下，lyrics 给 0.05
            w_meta   = 0.30
            w_attr   = 0.30
            w_bm25   = 0.15          # < 0.20
            w_lyrics = 0.05
            w_tag       = 0.12
            w_confirmed = 0.05
            w_pop       = 0.03

        else:
            # Case 4: default — lyrics 只给 0.05，其余保持 DeepSeek 动态权重
            w_lyrics = 0.05
            # 把 lyric_w 腾出来的空间补给 meta 和 attr
            freed = max(ss.lyric_w * VEC_TOTAL - 0.05, 0)
            w_meta = w_meta + freed * 0.6
            w_attr = w_attr + freed * 0.4

        # 否定实体合并
        rejected_tids = (set(query_dict.get("rejected_track_ids", []))
                       | set(ss.rejected_tracks))
        excl_artists  = (set(a.lower() for a in query_dict.get("excluded_artists", []))
                       | set(a.lower() for a in ss.rejected_artists))
        excl_tags     = set(t.lower() for t in ss.rejected_tags)

        # confirmed 合并
        confirmed = list(set(
            query_dict.get("confirmed_track_ids", []) + ss.confirmed_tracks
        ))

        # query_focus 覆盖 free_query
        if ss.query_focus:
            query_dict = dict(query_dict)
            query_dict["free_query"] = ss.query_focus

    else:
        confirmed     = query_dict.get("confirmed_track_ids", [])
        rejected_tids = set(query_dict.get("rejected_track_ids", []))
        excl_artists  = set(a.lower() for a in query_dict.get("excluded_artists", []))
        excl_tags     = set()

    excl_tids = set()

    # ── 封面意图检测：优先用 session_state，否则扫描 query 关键词 ──
    if session_state is not None and session_state.has_cover_intent:
        has_cover_intent = True
        # visual_description 优先于 free_query 用于封面路查询
        if session_state.visual_description:
            query_dict = dict(query_dict)
            query_dict["visual_description"] = session_state.visual_description
    else:
        free_q_lower = query_dict.get("free_query", "").lower()
        visual_desc  = query_dict.get("visual_description", "").lower()
        cover_text   = free_q_lower + " " + visual_desc
        has_cover_intent = any(kw in cover_text for kw in COVER_KEYWORDS)

    if has_cover_intent and track_indices.get("cover") is not None:
        # cover 路权重从 0.05 提升到 0.25，增量 0.20 从其他信号等比扣除
        COVER_W = 0.25
        delta = COVER_W - w_cover           # 0.25 - 0.05 = 0.20
        others_sum = w_meta + w_attr + w_lyrics + w_bm25 + w_tag + w_confirmed + w_pop
        scale = (others_sum - delta) / others_sum
        w_cover     = COVER_W
        w_meta      *= scale
        w_attr      *= scale
        w_lyrics    *= scale
        w_bm25      *= scale
        w_tag       *= scale
        w_confirmed *= scale
        w_pop       *= scale

    # ── 打印本次召回各路权重 ──
    if session_state is not None:
        if has_lyric_intent:    intent_case = "lyric"
        elif has_artist_intent: intent_case = "artist"
        elif has_scene_intent:  intent_case = "scene"
        else:                   intent_case = "default"
    else:
        intent_case = "no_state"
    if has_cover_intent:
        intent_case += "+cover"
    print(
        f"  [weights] meta={w_meta:.3f}  attr={w_attr:.3f}  lyrics={w_lyrics:.3f}  cover={w_cover:.3f}"
        f"  bm25={w_bm25:.3f}  tag={w_tag:.3f}  conf={w_confirmed:.3f}  pop={w_pop:.3f}"
        f"  | intent={intent_case}",
        flush=True,
    )

    # ── 解包四个索引 ──
    meta_ids,   meta_matrix   = track_indices["metadata"]
    attr_entry  = track_indices.get("attributes")
    lyr_entry   = track_indices.get("lyrics")
    cov_entry   = track_indices.get("cover")
    tid_to_meta_idx = {tid: i for i, tid in enumerate(meta_ids)}

    # ── Stage 1a: metadata cosine ANN ──
    q_vec_meta   = build_query_vector(query_dict, meta_ids, meta_matrix, confirmed, alpha)
    meta_scores  = meta_matrix @ q_vec_meta
    candidate_tids = set(
        meta_ids[i] for i in np.argsort(-meta_scores)[:recall_k]
    )

    # ── Stage 1b: attributes cosine ANN ──
    attr_scores_map: dict[str, float] = {}
    if attr_entry is not None:
        attr_ids, attr_matrix = attr_entry
        attr_free = ". ".join(filter(None, [
            " ".join(query_dict.get("mood_keywords", [])),
            query_dict.get("functional_context", ""),
            " ".join(query_dict.get("tags", [])),
        ]))
        if not attr_free.strip():
            attr_free = _build_free_text(query_dict)
        q_vec_attr = encode_queries([attr_free])[0]
        raw_attr   = attr_matrix @ q_vec_attr
        top_attr   = np.argsort(-raw_attr)[:recall_k]
        for i in top_attr:
            tid = attr_ids[i]
            attr_scores_map[tid] = float(raw_attr[i])
            candidate_tids.add(tid)

    # ── Stage 1c: lyrics cosine ANN ──
    lyr_scores_map: dict[str, float] = {}
    if lyr_entry is not None:
        lyr_ids, lyr_matrix = lyr_entry
        lyric_themes = query_dict.get("lyric_themes", [])
        lyr_free = (
            "Lyrics about: " + ", ".join(lyric_themes)
            if lyric_themes
            else query_dict.get("free_query", _build_free_text(query_dict))
        )
        q_vec_lyr = encode_queries([lyr_free])[0]
        raw_lyr   = lyr_matrix @ q_vec_lyr
        top_lyr   = np.argsort(-raw_lyr)[:recall_k]
        for i in top_lyr:
            tid = lyr_ids[i]
            lyr_scores_map[tid] = float(raw_lyr[i])
            candidate_tids.add(tid)

    # ── Stage 1d: cover cosine ANN（SigLIP2，仅封面意图时激活）──
    cov_scores_map: dict[str, float] = {}
    if has_cover_intent and cov_entry is not None:
        cov_ids, cov_matrix = cov_entry
        # 用 visual_description 优先，否则用 free_query 作为封面描述
        cover_desc = (
            query_dict.get("visual_description")
            or query_dict.get("free_query")
            or _build_free_text(query_dict)
        )
        q_vec_cov = encode_cover_query([cover_desc])[0]   # (768,) L2-normalized
        raw_cov   = cov_matrix @ q_vec_cov
        top_cov   = np.argsort(-raw_cov)[:recall_k]
        for i in top_cov:
            tid = cov_ids[i]
            cov_scores_map[tid] = float(raw_cov[i])
            candidate_tids.add(tid)

    # ── Stage 1e: BM25 ──
    bm25_query     = _build_bm25_query(query_dict)
    bm25_results   = bm25_index.search(bm25_query, top_k=recall_k)
    bm25_score_map = {tid: score for tid, score in bm25_results}
    candidate_tids |= set(bm25_score_map.keys())

    # ── Stage 1f: 否定过滤 ──
    candidate_indices = {
        tid_to_meta_idx[tid] for tid in candidate_tids if tid in tid_to_meta_idx
    }
    filtered_indices = _post_recall_filter(
        candidate_indices=candidate_indices,
        track_ids=meta_ids,
        track_meta=track_meta,
        rejected_track_ids=rejected_tids,
        excluded_artists=excl_artists,
        excluded_track_ids=excl_tids,
    )

    if not filtered_indices:
        return []

    conf_indices = [tid_to_meta_idx[tid] for tid in confirmed if tid in tid_to_meta_idx]

    # ── Stage 2: 七信号重排序 ──
    raw_candidates = []
    for idx in filtered_indices:
        tid  = meta_ids[idx]
        meta = track_meta.loc[tid] if tid in track_meta.index else None
        raw_candidates.append({
            "tid":    tid,
            "meta":   float(meta_scores[idx]),
            "attr":   attr_scores_map.get(tid, 0.0),
            "lyrics": lyr_scores_map.get(tid, 0.0),
            "cover":  cov_scores_map.get(tid, 0.0),
            "bm25":   bm25_score_map.get(tid, 0.0),
            "tag":    _tag_overlap_score(query_dict, meta) if meta is not None else 0.0,
            "pop":    _popularity_score(meta) if meta is not None else 0.0,
            "conf":   _confirmed_similarity_score(idx, meta_matrix, conf_indices),
        })

    # min-max 归一化
    for key in ("meta", "attr", "lyrics", "cover", "bm25", "tag", "pop", "conf"):
        vals  = np.array([c[key] for c in raw_candidates], dtype=np.float32)
        v_min, v_max = vals.min(), vals.max()
        span  = v_max - v_min
        for c in raw_candidates:
            c[key + "_n"] = (c[key] - v_min) / span if span > 1e-9 else 0.0

    for c in raw_candidates:
        c["final"] = (
            w_meta      * c["meta_n"]    +
            w_attr      * c["attr_n"]    +
            w_lyrics    * c["lyrics_n"]  +
            w_cover     * c["cover_n"]   +
            w_bm25      * c["bm25_n"]    +
            w_tag       * c["tag_n"]     +
            w_confirmed * c["conf_n"]    +
            w_pop       * c["pop_n"]
        )

    raw_candidates.sort(key=lambda x: -x["final"])
    return [c["tid"] for c in raw_candidates[:top_k]]