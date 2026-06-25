from flask import Flask, request, jsonify, render_template
import torch
import inference
import tokenizers
from model import Transformer
import config
import math
import sqlite3

def init_db():
    conn = sqlite3.connect("translations.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            english TEXT NOT NULL,
            french TEXT NOT NULL,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

en_tokenizer = tokenizers.Tokenizer.from_file("en_tokenizer.json")
fr_tokenizer = tokenizers.Tokenizer.from_file("fr_tokenizer.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
        num_layers     = config.NUM_LAYERS,
        num_heads      = config.NUM_HEADS,
        num_kv_heads   = config.NUM_KV_HEADS,
        hidden_dim     = config.HIDDEN_DIM,
        max_seq_len    = config.MAX_SEQ_LEN,
        vocab_size_src = len(en_tokenizer.get_vocab()),
        vocab_size_tgt = len(fr_tokenizer.get_vocab()),
        dropout        = config.DROPOUT,
    ).to(device)

model.load_state_dict(torch.load(config.CHECKPOINT, map_location=device, weights_only=True))
model.eval()
init_db()
print("[DB] Database initialised.")
print("[LOAD] Model ready.\n")

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    sentence = data["text"]
    # running greedy_decode here
    en_ids = torch.tensor(
            en_tokenizer.encode(sentence.lower().strip()).ids
        ).unsqueeze(0).to(device)
    print("Starting the beam search decoder")
    token_ids, normalised_score  = inference.beam_search_decode(
            model=model,
            en_ids=en_ids,
            en_tokenizer=en_tokenizer,
            fr_tokenizer=fr_tokenizer,
            device=device
        )
    translation = fr_tokenizer.decode(token_ids)
    confidence = round(math.exp(normalised_score) * 100, 1)
    conn = sqlite3.connect("translations.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO translations (english, french, confidence) VALUES(?, ?, ?)",
        (sentence, translation, confidence)
    )
    conn.commit()
    conn.close()
    return jsonify({"translation": translation, "confidence": confidence})

@app.route("/translate/candidates", methods=["POST"])
def translate_candidates():
    data = request.get_json()
    sentence = data["text"]
    en_ids = torch.tensor(
        en_tokenizer.encode(sentence.lower().strip()).ids
    ).unsqueeze(0).to(device)
    
    candidates = inference.beam_search_candidates(
        model=model,
        en_ids=en_ids,
        en_tokenizer=en_tokenizer,
        fr_tokenizer=fr_tokenizer,
        device=device
    )
    decoded = [fr_tokenizer.decode(ids) for ids in candidates]
    print(jsonify({"candidates": decoded}))
    return jsonify({"candidates": decoded})

@app.route("/history", methods = ["GET"])
def get_history():
    conn = sqlite3.connect("translations.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM translations ORDER BY timestamp  DESC LIMIT 20"
    )
    rows = cursor.fetchall()
    conn.close()
    return jsonify({"history":[
        {"english" : r[1], "french": r[2], "confidence":r[3], "timestamp":r[4]}
        for r in rows
    ]})

@app.route("/history-page",methods = ["GET"])
def history_page():
    return render_template("history.html")

@app.route("/api-docs")
def api_docs():
    return render_template("api_docs.html")

if __name__ == "__main__":
    app.run(debug=True)