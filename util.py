from torch.utils.data import Dataset

def read_qrels(min_rel = 4):
    qrels = []
    with open("data/qrels.qrels") as f:
        for line in f:
            q_id, doc_id, rel, _ = line.split()
            if int(rel) >= min_rel:
                qrels.append((q_id, doc_id))
    return qrels, set(qrel[1] for qrel in qrels)

def read_docs(kept_docs: set):
    docs = {}
    with open("data/docs.tsv") as f:
        for line in f:
            doc_id, doc_text = line.split("\t")
            if doc_id in kept_docs:
                docs[doc_id] = doc_text
    return docs

def read_queries():
    queries = {}
    with open("data/queries.tsv") as f:
        for line in f:
            q_id, q_text = line.split("\t")
            queries[q_id] = q_text
    return queries

class TRECDataset(Dataset):
    def __init__(self, min_rel=4):
        self.qrels, kept_docs = read_qrels(min_rel)
        self.docs = read_docs(kept_docs)
        self.queries = read_queries()

    def __len__(self):
        return len(self.qrels)

    def __getitem__(self, idx):
        qrel = self.qrels[idx]
        return self.queries[qrel[0]], self.docs[qrel[1]]

