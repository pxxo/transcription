from flask import Flask, request, jsonify, Response
import whisper
import os
import math
import tempfile
import numpy as np
import time
from flask_cors import CORS
import json
from sudachipy import tokenizer
from sudachipy import dictionary

app = Flask(__name__)
CORS(app)

models = {
    "large": whisper.load_model("large-v3", device="cuda"),
    "medium": whisper.load_model("medium", device="cuda"),
    "small": whisper.load_model("small", device="cuda"),
}


model = models["large"]
print("✅ モデル読み込み完了")

tokenizer_obj = dictionary.Dictionary().create()
split_mode = tokenizer.Tokenizer.SplitMode.C


def add_punctuation(text):
    text = text.strip()
    sentences = []
    start = 0
    for m in tokenizer_obj.tokenize(text, split_mode):
        if m.surface() in "。！？.!?":
            sentences.append(text[start : m.end()])
            start = m.end()
    if start < len(text):
        sentences.append(text[start:])
    # 各文末に句点がなければ追加
    result = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not s.endswith(("。", "！", "？", ".", "!", "?", "、")):
            # 文末の語を取得して、句読点の種類(「。」「、」)を決定
            last_token = tokenizer_obj.tokenize(s, split_mode)[-1]
            if last_token.surface() in [
                "です",
                "ます",
                "だ",
                "だった",
                "でした",
                "する",
                "した",
                "ですよ",
                "ますよ",
                "だよ",
                "だったよ",
                "でしたよ",
                "するよ",
                "したよ",
                "でしょう",
                "ますね",
                "だね",
                "だったね",
                "したね",
                "でしょうか",
                "ますか",
                "だか",
                "だったか",
                "でしたか",
                "するか",
                "したか",
                "でしょうね",
                "でしたね",
                "するね",
            ]:
                s += "。"
            elif last_token.surface() in [
                "が",
                "を",
                "に",
                "で",
                "と",
                "へ",
                "から",
                "まで",
                "より",
                "は",
                "も",
                "ね",
                "けど",
                "な",
                "して",
                "だ",
                "えて",
                "して",
                "たら",
            ]:
                s += "、"
            elif last_token.surface() in ["！", "！", "？", "？"]:
                s += last_token.surface()
        result += s
    return result


@app.route("/transcribe", methods=["POST"])
def transcribe_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        file.save(tmp.name)
        audio_path = tmp.name
        model_sr = 16000
        segment_sec = 30
        audio = whisper.load_audio(audio_path)
        audio = np.copy(audio)
        total_sec = audio.shape[-1] / model_sr
        num_segments = math.ceil(total_sec / segment_sec)

    start_time = time.time()

    def generate():
        chat_log = []
        for i in range(num_segments):
            start = int(i * segment_sec * model_sr)
            end = int(min((i + 1) * segment_sec * model_sr, audio.shape[-1]))
            audio_segment = audio[start:end]
            if audio_segment.shape[-1] < 1000:
                continue
            result = model.transcribe(
                audio_segment, language="ja", verbose=False, word_timestamps=True
            )
            seg_start_sec = i * segment_sec
            if "segments" in result and result["segments"]:
                for seg in result["segments"]:
                    s = seg.get("start", 0) + seg_start_sec
                    e = seg.get("end", 0) + seg_start_sec
                    text = add_punctuation(seg["text"].strip())
                    chat_log.append(
                        {
                            "start": float(f"{s:.2f}"),
                            "end": float(f"{e:.2f}"),
                            "text": text,
                        }
                    )
                    progress = (i + 1) / num_segments
                    yield json.dumps(
                        {"progress": progress, "result": chat_log[-1]}
                    ) + "\n"
            else:
                text = add_punctuation(result["text"].strip())
                chat_log.append(
                    {
                        "start": float(f"{seg_start_sec:.2f}"),
                        "end": float(f"{seg_start_sec+segment_sec:.2f}"),
                        "text": text,
                    }
                )
                progress = (i + 1) / num_segments
                yield json.dumps({"progress": progress, "result": chat_log[-1]}) + "\n"
        elapsed = time.time() - start_time
        try:
            time.sleep(0.1)
            os.remove(audio_path)
        except PermissionError:
            pass
        yield json.dumps({"done": True, "elapsed": elapsed}) + "\n"

    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
