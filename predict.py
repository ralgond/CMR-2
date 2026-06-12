"""
predict.py
"""
import argparse
import json
import os
import time

import pandas as pd
from tqdm import tqdm

from data_loader import (
    build_track_indices,
    get_user_history,
    load_blind_sessions,
    load_track_metadata,
    load_train_sessions,
)
from retriever import build_bm25_index, extract_query, retrieve
from session_state import accumulate_state_from_conversations, SessionState


def predict_session(
    session_row: pd.Series,
    track_indices: dict,
    track_meta: pd.DataFrame,
    train_df: pd.DataFrame,
    top_k: int = 20,
) -> dict:
    conversations = session_row["conversations"]
    session_id    = session_row["session_id"]
    user_id       = session_row["user_id"]

    last_turn = max(
        t["turn_number"] for t in conversations if t["role"] == "user"
    )

    state = accumulate_state_from_conversations(conversations, last_turn - 1)
    query_dict = extract_query(conversations, up_to_turn=last_turn)

    query_dict["confirmed_track_ids"] = list(set(
        state.confirmed_tracks + query_dict.get("confirmed_track_ids", [])
    ))
    query_dict["rejected_track_ids"] = list(set(
        state.rejected_tracks + query_dict.get("rejected_track_ids", [])
    ))

    profile = session_row.get("user_profile", {}) or {}
    culture = profile.get("preferred_musical_culture", "")
    if culture:
        tags = query_dict.setdefault("tags", [])
        if culture.lower() not in [t.lower() for t in tags]:
            tags.append(culture)
    lang = profile.get("preferred_language", "")
    if lang and not query_dict.get("language"):
        query_dict["language"] = lang

    user_history = get_user_history(user_id, train_df)

    predicted_ids = retrieve(
        query_dict=query_dict,
        track_indices=track_indices,
        track_meta=track_meta,
        user_history=user_history,
        session_state=state,
        top_k=top_k,
    )

    return {
        "session_id":          session_id,
        "user_id":             user_id,
        "turn_number":         last_turn,
        "predicted_track_ids": predicted_ids,
        "predicted_response":  "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",       default="submission.json")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()

    print("=" * 55)
    print(" TalkPlay Challenge — predict.py")
    print("=" * 55)

    print("\n[1/4] Loading data...")
    blind_df   = load_blind_sessions()
    train_df   = load_train_sessions()
    track_meta = load_track_metadata()
    print(f"      Blind sessions : {len(blind_df)}")
    print(f"      Train sessions : {len(train_df)}")
    print(f"      Recall pool    : {len(track_meta)} tracks")

    print("\n[2/4] Building BM25 index...")
    build_bm25_index(track_meta)

    print("\n[3/4] Building track indices (metadata / attributes / lyrics)...")
    track_indices = build_track_indices()

    print(f"\n[4/4] Predicting...")
    sessions = blind_df.head(args.max_sessions) if args.max_sessions else blind_df

    all_results = []
    for _, row in tqdm(sessions.iterrows(), total=len(sessions), desc="Sessions"):
        try:
            result = predict_session(
                session_row=row,
                track_indices=track_indices,
                track_meta=track_meta,
                train_df=train_df,
            )
        except Exception as exc:
            sid = row["session_id"]
            last_turn = max(
                t["turn_number"] for t in row["conversations"] if t["role"] == "user"
            )
            print(f"  [WARN] session={sid[:8]}: {exc}")
            result = {
                "session_id":          sid,
                "user_id":             row["user_id"],
                "turn_number":         last_turn,
                "predicted_track_ids": [],
                "predicted_response":  "",
            }
        all_results.append(result)
        time.sleep(0.25)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(all_results)} predictions -> {args.output}")


if __name__ == "__main__":
    main()