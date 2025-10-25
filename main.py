    import collections
    import sys
    import os
    import re
    import traceback
    import threading
    import time
    import requests

    # FIX FOR sumy IN PYTHON 3.12
    if not hasattr(collections, 'Sequence'):
        collections.Sequence = collections.abc.Sequence

    from flask import Flask, render_template, request, jsonify
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer

    try:
        from rake_nltk import Rake
    except Exception:
        Rake = None

    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        print("Warning: nltk.download('punkt') failed:", e)

    app = Flask(__name__, static_folder="static", template_folder="templates")


    def count_words(text):
        """Count the number of words in a text string"""
        if not text:
            return 0
        words = text.split()
        return len(words)


    def extract_main_topic(text):
        if not Rake:
            return "This topic"
        try:
            rake = Rake()
            rake.extract_keywords_from_text(text)
            phrases = rake.get_ranked_phrases()
            if phrases:
                return phrases[0].strip().capitalize()
        except Exception:
            pass
        return "This topic"


    def highlight_keywords(summary, top_n=5):
        if not Rake:
            return summary
        try:
            rake = Rake()
            rake.extract_keywords_from_text(summary)
            keywords = rake.get_ranked_phrases()[:top_n]
            keywords = sorted(keywords, key=lambda s: -len(s))
            for kw in keywords:
                if not kw.strip():
                    continue
                pattern = re.compile(r"\b" + re.escape(kw) + r"\b", flags=re.IGNORECASE)
                summary = pattern.sub(lambda m: f"<strong>{m.group(0)}</strong>", summary)
        except Exception:
            return summary
        return summary


    def clean_first_sentence(sentences_list, topic):
        if not sentences_list:
            return sentences_list
        first = sentences_list[0].strip()
        if not first:
            return sentences_list
        first_word = first.split()[0].lower().rstrip(".,;:")
        vague_starts = {
            "it",
            "this",
            "they",
            "these",
            "those",
            "despite",
            "however",
            "but",
            "although",
            "though",
            "while",
            "yet",
        }
        if first_word in vague_starts:
            new_first = (
                f"{topic}, {first[0].lower()}{first[1:]}"
                if first and first[0].isupper()
                else f"{topic}, {first}"
            )
            new_first = new_first[0].upper() + new_first[1:]
            sentences_list[0] = new_first
        return sentences_list


    def raw_sentence_split(text):
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
            if sents:
                return sents
        except Exception:
            pass
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if s.strip()]


    @app.route("/")
    def home():
        return render_template("index.html")


    @app.route("/health")
    def health():
        return jsonify({"status": "ok"})


    @app.route("/summarize", methods=["POST"])
    def summarize():
        try:
            data = request.get_json(silent=True) or request.form or {}
            text = (data.get("paragraph") or data.get("text") or "").strip()
            if not text:
                return jsonify({"error": "No text provided", "summary": ""}), 400

            # Count words in original text
            original_word_count = count_words(text)

            try:
                sentences_requested = int(data.get("sentences", 3))
                if sentences_requested < 1:
                    sentences_requested = 3
            except Exception:
                sentences_requested = 3

            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()

            all_sent_objs = list(parser.document.sentences)
            all_sent_texts = [str(s).strip() for s in all_sent_objs if str(s).strip()]

            summary_sentences = []

            if all_sent_texts:
                summary_sentences.append(all_sent_texts[0])

            if sentences_requested > 1:
                try:
                    ranked = list(summarizer(parser.document, sentences_count=max(5, sentences_requested + 3)))
                    for s in ranked:
                        s_text = str(s).strip()
                        if not s_text:
                            continue
                        if all_sent_texts and s_text == all_sent_texts[0]:
                            continue
                        if s_text in summary_sentences:
                            continue
                        summary_sentences.append(s_text)
                        if len(summary_sentences) >= sentences_requested:
                            break
                except Exception:
                    pass

            if len(summary_sentences) < sentences_requested and len(all_sent_texts) > 1:
                for s in all_sent_texts[1:]:
                    if s not in summary_sentences:
                        summary_sentences.append(s)
                    if len(summary_sentences) >= sentences_requested:
                        break

            if not summary_sentences:
                raw_sents = raw_sentence_split(text)
                summary_sentences = raw_sents[:sentences_requested]

            topic = extract_main_topic(text)
            summary_sentences = clean_first_sentence(summary_sentences, topic)

            summary = " ".join(summary_sentences).strip()

            if not summary:
                words = text.split()
                summary = " ".join(words[:min(40, len(words))]) + ("..." if len(words) > 40 else "")

            summary = highlight_keywords(summary, top_n=5)

            # Count words in summary
            summary_word_count = count_words(summary)

            # Calculate reduction percentage
            if original_word_count > 0:
                reduction_percentage = round(((original_word_count - summary_word_count) / original_word_count) * 100)
            else:
                reduction_percentage = 0

            # ** ADD WARNING MESSAGE HERE **
            warning_msg = "<p style='color:#ff9800; font-weight:bold;'>⚠️ Summarizer can make mistakes, summarize sensitive or important content at your own risk.</p>"
            summary = warning_msg + summary

            return jsonify({
                "summary": summary,
                "original_word_count": original_word_count,
                "summary_word_count": summary_word_count,
                "reduction_percentage": reduction_percentage
            })

        except Exception as e:
            print("Exception in /summarize:", e)
            traceback.print_exc()
            return jsonify({"error": "Internal server error", "summary": ""}), 500


    # -------------------------
    # --- Keep Alive Code ---
    def keep_alive():
        def run():
            while True:
                try:
                    # Replace with your real Replit link
                    url = "https://7adfb29d-7171-4ef4-83ec-af6016f91ba9-00-27ysa52o0o5yb.pike.replit.dev/"
                    requests.get(url)
                    print("Pinged self to stay awake.")
                except Exception as e:
                    print("Ping failed:", e)
                time.sleep(280)  # ~5 minutes

        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()


    # -------------------------
    # --- Flask Run ---
    if __name__ == "__main__":
        keep_alive()   # <--- Start the self-pinger
        port = int(os.environ.get("PORT", 8000))
        print(f"Starting server on 0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=True)