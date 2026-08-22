import argparse, json, os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

FINAL_LABELS = ["sangat positif","positif","netral","negatif","sangat negatif"]

def load_texts(path: str):
    lp = path.lower()
    if lp.endswith(".csv"):
        df = pd.read_csv(path)
        if "text" not in df.columns:
            raise ValueError("CSV harus punya kolom 'text'")
        return df["text"].astype(str).tolist()

    if lp.endswith(".jsonl"):
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                texts.append(str(obj["text"]))
        return texts

    raise ValueError("input harus .csv atau .jsonl")

def choose_label_5_from_3(pos, neu, neg, pos_strong, neg_strong):
    if pos >= neu and pos >= neg:
        return "sangat positif" if pos >= pos_strong else "positif"
    if neg >= neu and neg >= pos:
        return "sangat negatif" if neg >= neg_strong else "negatif"
    return "netral"

def build_scores5(label5, pos, neu, neg):
    scores5 = {k: 0.0 for k in FINAL_LABELS}
    if label5 == "netral":
        scores5["netral"] = float(neu)
    elif label5 == "positif":
        scores5["positif"] = float(pos)
    elif label5 == "sangat positif":
        scores5["sangat positif"] = float(pos)
    elif label5 == "negatif":
        scores5["negatif"] = float(neg)
    elif label5 == "sangat negatif":
        scores5["sangat negatif"] = float(neg)
    return scores5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pos_strong", type=float, default=0.7)
    ap.add_argument("--neg_strong", type=float, default=0.7)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    texts = load_texts(args.input)

    # id2label bisa berupa: {0:'negative',1:'neutral',2:'positive'} atau variasi
    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}

    results = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, truncation=True, padding=True, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs_batch = torch.softmax(logits, dim=-1).tolist()

            for j, probs in enumerate(probs_batch):
                t = batch_texts[j]
                probs3 = {}
                for idx, p in enumerate(probs):
                    lab = id2label.get(idx, str(idx)).lower()
                    probs3[lab] = float(p)

                # normalisasi key label 3 kelas
                pos = probs3.get("positive", probs3.get("positif", 0.0))
                neu = probs3.get("neutral", probs3.get("netral", probs3.get("neu", 0.0)))
                neg = probs3.get("negative", probs3.get("negatif", 0.0))

                # kalau model punya urutan label lain (jarang), fallback: ambil top-3 dan klasifikasi berdasarkan kata
                if (pos == 0.0 and neg == 0.0 and neu == 0.0) and len(probs3) > 0:
                    pos = 0.0
                    neg = 0.0
                    neu = 0.0
                    for lab, p in probs3.items():
                        if "pos" in lab:
                            pos = p
                        elif "neg" in lab:
                            neg = p
                        elif "neu" in lab or "neutral" in lab or "netral" in lab:
                            neu = p

                label5 = choose_label_5_from_3(pos, neu, neg, args.pos_strong, args.neg_strong)
                scores5 = build_scores5(label5, pos, neu, neg)

                results.append({
                    "text": t,
                    "label": label5,
                    "scores": scores5
                })

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

