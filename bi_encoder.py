from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiEncoder(nn.Module):
    def __init__(
        self, model_name="sentence-transformers/all-MiniLM-L6-v2", temperature=0.1
    ):
        super(BiEncoder, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.item_encoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature

    def _mean_pool(self, model_output, attention_mask):
        token_embs = model_output.last_hidden_state  # (B, seq_len, D)
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
        return (token_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

    def _encode(self, encoder, texts):
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(next(encoder.parameters()).device) for k, v in inputs.items()}
        output = encoder(**inputs)
        embs = self._mean_pool(output, inputs["attention_mask"])
        return F.normalize(embs, p=2, dim=-1)

    def encode_query(self, query):
        return self._encode(self.query_encoder, query)

    def encode_item(self, item):
        return self._encode(self.item_encoder, item)

    def forward(self, queries, documents):
        """
        Args:
            queries: list of query strings (batch_size)
            documents: list of positive document strings (batch_size)
                       queries[i] is paired with documents[i]
        Returns:
            query_embs: (B, D) normalized query embeddings
            doc_embs: (B, D) normalized document embeddings
        """
        query_embs = self._encode(self.query_encoder, queries)
        doc_embs = self._encode(self.item_encoder, documents)
        return query_embs, doc_embs

    def contrastive_loss(self, query_embs, doc_embs):
        """
        InfoNCE loss using in-batch negatives.

        query_embs: (B, D) — one query per row
        doc_embs:   (B, D) — one positive doc per row

        The similarity matrix is (B, B) where entry [i,j] = sim(Q_i, D_j).
        The target is the diagonal (each query's positive is at index i).
        """
        # (B, B) query-document similarity scaled by temperature
        qd_scores = torch.matmul(query_embs, doc_embs.T) / self.temperature

        labels = torch.arange(qd_scores.size(0), device=qd_scores.device)
        return F.cross_entropy(
            qd_scores, labels
        )  # diagonals are positive targets, should get highest p

    def distillation_loss(self, query_embs, doc_embs):
        """
        KL-divergence between query-document similarity distribution and
        document-document similarity distribution (soft labels).

        For each query Q_i with positive D_i:
          - teacher distribution: softmax of row i in the D-D similarity matrix
            (how similar D_i is to every document in the batch)
          - student distribution: softmax of row i in the Q-D similarity matrix
            (how the model currently scores Q_i against every document)

        This penalizes the model for pushing away documents that the positive
        document considers similar.
        """
        # (B, B) document-document similarity — teacher signal
        dd_scores = torch.matmul(doc_embs, doc_embs.T) / self.temperature
        # (B, B) query-document similarity — student signal
        qd_scores = torch.matmul(query_embs, doc_embs.T) / self.temperature

        teacher_dist = (
            F.softmax(dd_scores, dim=-1).detach()
        )  # Use model itself as the teacher to generate soft labels, detach gradient
        student_log_dist = F.log_softmax(qd_scores, dim=-1)

        return F.kl_div(student_log_dist, teacher_dist, reduction="batchmean")

    def loss(self, query_embs, doc_embs, alpha=1.0, beta=1.0):
        """
        Combined loss = contrastive + alpha * distillation.

        Args:
            query_embs: (B, D) normalized query embeddings
            doc_embs:   (B, D) normalized document embeddings
            alpha: weight for the InfoNCE term
            beta: weight for the distiallation term
        """
        cl = self.contrastive_loss(query_embs, doc_embs)
        dl = self.distillation_loss(query_embs, doc_embs)
        return (
            alpha * cl + beta * dl,
            cl,
            dl,
        )  # for some reason cl and dl are almost the same, at least with BGE it was
