import os
import json
from openai import OpenAI
import pandas as pd

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-api-key"),
    base_url="https://api.deepseek.com",
)
DEEPSEEK_MODEL = "deepseek-chat"

SYSTEM_PROMPT='''
You are a music session state analyzer for a conversational music recommendation system.



You will receive a full conversation (user messages). In the conversation, there are some turns, you should process from turn 1, and while processing turn N, you should consider the output of turn N-1 (artist, tracks, tags), and merge the result of  N and N-1 (intersection or union or overwrite).
Analyze it turn by turn and extract the following structured state as JSON.

Output JSON schema:
{
    turn_op: [],
    
    "pos_tags": [],
    "pos_artists": [],
    "pos_tracks": [],
    
    "neg_tags": [],
    "neg_artists": [],
    "neg_tracks": [],
}

Field rules:

pos_artists: artists the user explicitly praised ("I love X", "from X", "more from X", "X is great").
pos_tracks: track UUIDs from music turns where the next user message was clearly positive.
pos_tags: styles/genres/moods/era/vibe the user said they want ("slow", "vocals", "jazzy", "80s", "2000s", "calming").
neg_artists: artists the user explicitly rejected ("not this artist", "heard enough from X", "no more X").
neg_tracks: track UUIDs (36-char strings) from music turns that the user rejected.
neg_tags: styles/genres/moods/vibe the user said they don't want ("not too slow", "no vocals", "not jazzy", "not calming").
turn_op: debug information show how to merge output of N-1 and N.


Only output valid JSON, no explanation, no markdown.
'''

class IntentParser:
    def __init__(self):
        pass

    def __call__(self, turn_l):
        user_turn_l = []
        for turn in turn_l:
            if turn['role'] == 'user':
                user_turn_l.append(turn['content'])

        user_turn_l2 = []
        for idx, user_turn in enumerate(user_turn_l, 1):
            user_turn_l2.append(f"Turn {idx} | {user_turn}")

        conv_text = '\n'.join(user_turn_l2)
        print(conv_text)
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
            return raw
        except Exception as e:
            print(f"  [session_state] DeepSeek error: {e}, using defaults")


def test():
    df = pd.read_parquet('data/Challenge-Data/train-00000-of-00001.parquet')
    
    session_data = df['conversations'].tolist()[0]
    
    ip = IntentParser()

    ip(session_data.tolist())



if __name__ == "__main__":
    test()