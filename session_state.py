"""
session_state.py
DeepSeek-powered session state analyzer.

将 turn 1~N 的完整对话打包发给 DeepSeek，逐 turn 分析后输出结构化状态。
"""

import json
import os
from dataclasses import dataclass, field
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key"),
    base_url="https://api.deepseek.com",
)
DEEPSEEK_MODEL = "deepseek-chat"


@dataclass
class SessionState:
    # ── 1. 三路召回权重（合计为 1.0，由 DeepSeek 动态分配）──
    meta_w: float = 0.35      # 具体歌手/曲目导向（metadata cosine + BM25）
    attr_w: float = 0.40      # 场景/情绪/属性导向（attributes cosine）
    lyric_w: float = 0.25     # 抽象歌词/主题导向（lyrics cosine）

    # ── 2. 否定实体 ──
    rejected_artists: list[str] = field(default_factory=list)   # 不想要的艺术家
    rejected_tracks: list[str] = field(default_factory=list)    # 不想要的具体曲目（UUID）
    rejected_tags: list[str] = field(default_factory=list)      # 不想要的风格/标签

    # ── 3. 保持风格但换艺术家 ──
    keep_style_new_artist: bool = False   # 用户说"这个风格好但换个艺术家"
    style_anchors: list[str] = field(default_factory=list)   # 当前风格锚点词

    # ── 4. 正向艺术家信号（补充）──
    confirmed_artists: list[str] = field(default_factory=list)  # 用户明确喜欢的艺术家
    confirmed_tracks: list[str] = field(default_factory=list)   # 确认喜欢的曲目 UUID

    # ── 5. 意图迁移检测（补充）──
    session_shift_detected: bool = False
    shift_summary: str = ""   # 迁移描述，e.g. "从找特定艺术家转向找放松氛围"

    # ── 6. 当前核心诉求（补充）──
    query_focus: str = ""     # 最后一个 user turn 的核心诉求（一句话）

    # ── 7. 具体程度（补充）──
    specificity_level: str = "mid"  # high | mid | low

    # ── 8. 封面意图 ──
    has_cover_intent: bool = False       # 用户提到封面相关词汇
    visual_description: str = ""         # 用户对封面的具体描述（颜色/形状/人物/风格等）


SYSTEM_PROMPT = """
You are a music session state analyzer for a conversational music recommendation system.

You will receive a full conversation (user messages + music recommendations).
Analyze it turn by turn and extract the following structured state as JSON.

Output JSON schema:
{
    'turn1':{
        "rejected_artists": [],
        "rejected_tracks": [],
        "rejected_tags": [],
        
        "confirmed_artists": [],
        "confirmed_tracks": [],
        "confirmed_tags": [],
        
        "session_shift_detected": false,
        "shift_summary": "",
        
        "query_focus": "",
        
        "has_cover_intent": false,
        "visual_description": ""
    },
    'turn2':{
        
    },
    ...
}

Field rules:

confirmed_artists: artists the user explicitly praised ("I love X", "more from X", "X is great")
confirmed_tracks: track UUIDs from music turns where the next user message was clearly positive
confirmed_tags: styles/genres/moods the user said they want ("slow", "vocals", "jazzy")

rejected_artists: artists the user explicitly rejected ("not this artist", "heard enough from X", "no more X")
rejected_tracks: track UUIDs (36-char strings) from music turns that the user rejected
rejected_tags: styles/genres/moods the user said they don't want ("not too slow", "no vocals", "not jazzy")

session_shift_detected: true if user's goal changed significantly across turns
  (e.g. started looking for workout music, then shifted to asking for a specific artist)
shift_summary: one sentence describing what changed, empty string if no shift

query_focus: one concise sentence summarizing what the user wants RIGHT NOW (last turn intent)
  Must be self-contained. Include style, mood, artist hints if present.

has_cover_intent: true if the user mentions album cover, cover art, artwork, sleeve, visual design,
  or describes what an album looks like. Keywords: "cover", "album cover", "album art", "artwork",
  "sleeve", "visual", "jacket", "what does the album look like", "cover image".

visual_description: extract the user's description of the cover visual — colors, shapes, people,
  objects, art style, mood of the image. E.g. "skull on dark background", "black and white portrait
  of a woman", "vibrant abstract colors", "person on a skateboard mid-air in a skate park".
  Empty string if has_cover_intent is false.

Track UUIDs appear as the entire content of music turns (role=music).
Only output valid JSON, no explanation, no markdown.
"""


def analyze_session(conversations: list[dict], up_to_turn: int) -> SessionState:
    """
    将 turn 1~up_to_turn 的完整对话发给 DeepSeek，
    分析后返回结构化 SessionState。
    """
    relevant = [t for t in conversations if t["turn_number"] <= up_to_turn]

    # 格式化对话，清晰标注每个 turn
    lines = []
    for t in relevant:
        role_label = {
            "user": "USER",
            "music": "MUSIC_REC",
            "assistant": "ASSISTANT",
        }.get(t["role"], t["role"].upper())
        content = t["content"][:500]  # 截断过长内容
        lines.append(f"[Turn {t['turn_number']} | {role_label}]: {content}")

    conv_text = "\n".join(lines)

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": conv_text},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"  [session_state] DeepSeek error: {e}, using defaults")
        return SessionState()

    # 权重归一化（防止 DeepSeek 输出不合法）
    meta_w  = float(raw.get("meta_w", 0.35))
    attr_w  = float(raw.get("attr_w", 0.40))
    lyric_w = float(raw.get("lyric_w", 0.25))
    total = meta_w + attr_w + lyric_w
    if total > 0:
        meta_w, attr_w, lyric_w = meta_w/total, attr_w/total, lyric_w/total
    else:
        meta_w, attr_w, lyric_w = 0.35, 0.40, 0.25

    return SessionState(
        meta_w=round(meta_w, 3),
        attr_w=round(attr_w, 3),
        lyric_w=round(lyric_w, 3),

        rejected_artists=raw.get("rejected_artists", []),
        rejected_tracks=raw.get("rejected_tracks", []),
        rejected_tags=raw.get("rejected_tags", []),

        keep_style_new_artist=bool(raw.get("keep_style_new_artist", False)),
        style_anchors=raw.get("style_anchors", []),

        confirmed_artists=raw.get("confirmed_artists", []),
        confirmed_tracks=raw.get("confirmed_tracks", []),

        session_shift_detected=bool(raw.get("session_shift_detected", False)),
        shift_summary=raw.get("shift_summary", ""),

        query_focus=raw.get("query_focus", ""),

        specificity_level=raw.get("specificity_level", "mid"),

        has_cover_intent=bool(raw.get("has_cover_intent", False)),
        visual_description=raw.get("visual_description", ""),
    )


def accumulate_state_from_conversations(
    conversations: list[dict], up_to_turn: int
) -> SessionState:
    """
    向后兼容接口（供 predict.py / validate.py 调用）。
    直接调用 DeepSeek 分析，替换旧的启发式规则。
    """
    if up_to_turn < 1:
        return SessionState()
    return analyze_session(conversations, up_to_turn)