from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

OUTPUT_DIR = Path("/data2/fyb/knndiffu/raco/faiss_index")
DATA_PATH = Path("/data2/fyb/knndiffu/raco/CommonsenseCorpus/Commonsense20M.tsv")
EMB_PATH = OUTPUT_DIR / "commonsense20M_embeddings.npy"
IDS_PATH = OUTPUT_DIR / "commonsense20M_ids.txt"
INDEX_PATH = OUTPUT_DIR / "raco.IndexFlatIP.faiss"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("***** Loading model ... *****\n")
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device_map": "auto"},
    tokenizer_kwargs={"padding_side": "left"},
)

if EMB_PATH.exists() and IDS_PATH.exists():
    print(f"***** Reusing cached embeddings from {EMB_PATH} *****\n")
    embeddings = np.load(EMB_PATH)
    with IDS_PATH.open("r", encoding="utf-8") as id_file:
        ids = [line.rstrip("\n") for line in id_file]
    print(f"***** Loaded {len(ids)} ids; embeddings shape: {embeddings.shape} *****\n")
else:
    print(f"***** Loading data from {DATA_PATH} ... *****\n")
    ids = []
    texts = []
    with DATA_PATH.open("r", encoding="utf-8") as source:
        for raw_line in source:
            stripped = raw_line.rstrip("\n")
            if not stripped:
                continue
            parts = stripped.split("\t")
            if not parts or not parts[0]:
                continue
            doc_id = parts[0]
            doc_text = "\t".join(parts[1:]).strip()
            ids.append(doc_id)
            texts.append(doc_text)

    if not ids:
        raise RuntimeError(f"No rows read from {DATA_PATH}; cannot build embeddings.")

    print(f"***** Data Loaded ! Total {len(texts)} samples *****\n")
    print(f"***** Example id: {ids[0]} *****\n")

    print("***** Computing embeddings... *****\n")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=100,
    )

    print("***** Embedding Computing Finished ! embedding shape:", embeddings.shape)

    print(f"***** Saving embeddings to {EMB_PATH} *****\n")
    np.save(EMB_PATH, embeddings)

    print(f"***** Saving ids to {IDS_PATH} *****\n")
    with IDS_PATH.open("w", encoding="utf-8") as id_file:
        for doc_id in ids:
            id_file.write(doc_id + "\n")

if embeddings.shape[0] != len(ids):
    raise RuntimeError(
        f"Embedding count ({embeddings.shape[0]}) does not match id count ({len(ids)})."
    )

print("***** Normalizing embeddings for cosine similarity *****\n")
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
doc_ids = np.arange(len(ids), dtype=np.int64)
index = faiss.IndexIDMap(index)
index.add_with_ids(embeddings, doc_ids)
print(f"***** Faiss index built ! Total {index.ntotal} vectors *****\n")

print(f"***** Saving faiss index to {INDEX_PATH} *****\n")
faiss.write_index(index, str(INDEX_PATH))
