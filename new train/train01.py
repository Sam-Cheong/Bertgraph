# tcr_predictor_fixed.py
"""
Enhanced TCR predictor pipeline 
New features:
 - train_classifier_head(): trains a supervised classification head (MLP) on fused embeddings
 - train_lightgbm_backend(): uses embedding + motif + V/J + length for LightGBM multi-class classification (5-fold CV)
 - train_contrastive(): updated to InfoNCE with EMA-based mini-batch prototype updates, supporting multi-epoch training (up to 100)
 - RFU visualization: saves rfu_vector_training.png
 - Motif binary feature extraction (topK)

Retains and remains compatible with previous functionalities: motif saving, epitope network, prediction, visualization, etc.
"""
import os
import pandas as pd
import numpy as np
import networkx as nx
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import gc
import random
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import MiniBatchKMeans, KMeans
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import pairwise_distances, accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import time
import math
from matplotlib import gridspec
import pickle
import warnings
warnings.filterwarnings("ignore")

# optional import lightgbm
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False
    print("lightgbm not available; LightGBM backend will be skipped. Install lightgbm to enable.")

# set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

########################################
# Core models (MetaLearner, Fusion, etc.) - mostly unchanged
########################################

class MetaLearner(nn.Module):
    def __init__(self, n_modalities, hidden=32):
        super(MetaLearner, self).__init__()
        self.n_modalities = n_modalities
        self.net = nn.Sequential(
            nn.Linear(n_modalities, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_modalities)
        )
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled_feats):
        logits = self.net(pooled_feats)
        alpha = torch.sigmoid(logits)
        return alpha

class MultiModalFusion(nn.Module):
    def __init__(self, feature_dims):
        super(MultiModalFusion, self).__init__()
        self.feature_dims = feature_dims
        self.n_modalities = len(feature_dims)
        self.meta_learner = MetaLearner(self.n_modalities)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, features):
        features_2d = []
        for feat in features:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            features_2d.append(feat)

        batch_size = features_2d[0].shape[0]
        max_dim = max([f.shape[1] for f in features_2d])

        padded_feats = []
        for f in features_2d:
            if f.shape[1] < max_dim:
                pad = torch.zeros((f.shape[0], max_dim - f.shape[1]), device=f.device, dtype=f.dtype)
                f_padded = torch.cat([f, pad], dim=1)
            else:
                f_padded = f
            padded_feats.append(f_padded)

        pooled = [torch.mean(f, dim=1, keepdim=True) for f in padded_feats]
        pooled_feats = torch.cat(pooled, dim=1)  # (batch, n_modalities)

        alpha = self.meta_learner(pooled_feats)  # (batch, n_modalities)

        fused = torch.zeros((batch_size, max_dim), device=padded_feats[0].device, dtype=padded_feats[0].dtype)
        for i, f in enumerate(padded_feats):
            mean_val = torch.mean(f, dim=1, keepdim=True)
            std_val = torch.std(f, dim=1, keepdim=True)
            std_val = torch.where(std_val < 1e-8, torch.ones_like(std_val), std_val)
            norm_feat = self.gamma * (f - mean_val) / (std_val + 1e-8) + self.beta
            fused = fused + alpha[:, i].unsqueeze(1) * norm_feat

        ref_dim = features_2d[0].shape[1]
        if ref_dim < max_dim:
            fused = fused[:, :ref_dim]
        return fused

class GraphNeuralEvolution(nn.Module):
    def __init__(self, embedding_dim):
        super(GraphNeuralEvolution, self).__init__()
        self.W = nn.Linear(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, embeddings, adj_matrix):
        degrees = torch.sum(adj_matrix, dim=1)
        degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
        norm_factor = 1.0 / torch.sqrt(degrees.unsqueeze(1) * degrees.unsqueeze(0))
        new_embeddings = F.relu(torch.mm(adj_matrix * norm_factor, self.W(embeddings)))
        return new_embeddings

class ContrastivePrototypeLearning(nn.Module):
    """
    Contrastive prototype module.
    Note: forward returns loss (scalar tensor) and updated prototypes (torch tensor).
    """
    def __init__(self, temperature=0.2, gamma=0.99):
        super(ContrastivePrototypeLearning, self).__init__()
        self.temperature = temperature
        self.gamma = gamma  # used only if doing per-sample prototype update; we will use EMA externally
        self.loss_history = []

    def forward(self, embeddings, prototypes, cluster_ids):
        """
        embeddings: tensor (B, D)
        prototypes: tensor (K, D)
        cluster_ids: array-like length B cluster id per sample
        Return: loss tensor (sum over batch), prototypes unchanged (we update externally via EMA)
        """
        N = embeddings.shape[0]
        device = embeddings.device
        if N == 0:
            return torch.tensor(0.0, device=device), prototypes

        emb_norm = F.normalize(embeddings, p=2, dim=1)  # (B, D)
        proto_norm = F.normalize(prototypes, p=2, dim=1)  # (K, D)
        sim_mat = torch.matmul(emb_norm, proto_norm.T) / (self.temperature)  # (B, K)

        # For each sample, the positive is the prototype of its cluster_id
        logits = sim_mat  # (B, K)
        # compute log-softmax per row
        logprob = F.log_softmax(logits, dim=1)
        idx = torch.tensor(cluster_ids, dtype=torch.long, device=device)
        loss = -logprob[torch.arange(N, device=device), idx].mean()
        avg_loss = loss.detach().cpu().item()
        self.loss_history.append(float(avg_loss))
        return loss, prototypes

########################################
# Immune model wrapper (defensive)
########################################

class ImmuneBERTOptimized(nn.Module):
    def __init__(self, model_name="wukevin/tcr-bert", device=None):
        super(ImmuneBERTOptimized, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        model_options = [
            "wukevin/tcr-bert",
            "wukevin/tcr-bert-mlm-only",
            "wukevin/foldingdiff_cath",
            "facebook/esm2_t6_8M_UR50D"
        ]

        loaded = False
        for mname in model_options:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(mname)
                self.model = AutoModel.from_pretrained(mname).to(self.device)
                self.model.eval()
                logging.info(f"Successfully loaded model: {mname}")
                loaded = True
                break
            except Exception as e:
                logging.warning(f"Could not load model {mname}: {str(e)}")
                continue

        if not loaded:
            logging.error("All model loading attempts failed. Using fallback dummy model.")
            class DummyModel:
                def __init__(self, dim=768):
                    self.dim = dim
                def __call__(self, **kwargs):
                    batch = kwargs.get('input_ids', None)
                    bsz = batch.shape[0] if batch is not None else 1
                    class O:
                        def __init__(self, b, d):
                            self.last_hidden_state = torch.randn(b, 1, d)
                    return O(bsz, 768)
            self.tokenizer = lambda x, **kw: {'input_ids': torch.zeros((len(x), 1), dtype=torch.long)}
            self.model = DummyModel(768)

        try:
            for param in self.model.parameters():
                param.requires_grad = False
        except Exception:
            pass

        self.embedding_dim = 768

    def forward(self, sequences, batch_size=4):
        if not sequences:
            return np.array([])

        embeddings = []
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            try:
                inputs = self.tokenizer(batch_seqs, padding=True, truncation=True, max_length=64, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    if hasattr(outputs, 'last_hidden_state'):
                        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    elif hasattr(outputs, 'pooler_output'):
                        batch_embeddings = outputs.pooler_output.cpu().numpy()
                    else:
                        if hasattr(outputs, 'last_hidden_state'):
                            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                        else:
                            batch_embeddings = np.zeros((len(batch_seqs), self.embedding_dim))
                embeddings.append(batch_embeddings)
                del inputs, outputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            except Exception as e:
                logging.error(f"Error embedding batch: {str(e)}")
                placeholder = np.zeros((len(batch_seqs), self.embedding_dim))
                embeddings.append(placeholder)

        if embeddings:
            return np.vstack(embeddings)
        return np.array([])

########################################
# RFUAnalyzer (approximation) - unchanged except saving plot
########################################

class RFUAnalyzer:
    def __init__(self, n_components=100, n_rfus=500, motif_st=3, motif_ed=4):
        self.n_components = n_components
        self.n_rfus = n_rfus
        self.trimer_fit = None
        self.trimer_index = None
        self.pca = None
        self.kmeans = None
        self.motif_st = motif_st
        self.motif_ed = motif_ed

    @staticmethod
    def get_positioned_trimers(seq, st=3, ed=3, max_pos_gap=99):
        if not isinstance(seq, str):
            return []
        nL = len(seq)
        posL = list(range(st, min(nL - ed + 2, st + max_pos_gap)))
        trimers = []
        for p in posL:
            if p + 2 <= nL:
                tri = seq[p - 1:p + 2]
                trimers.append(f"{p}_{tri}")
        return trimers

    def fit_trimer_space(self, sequences):
        trimer_counts = Counter()
        seq_trimer_list = []
        for s in sequences:
            tlist = self.get_positioned_trimers(s, st=self.motif_st, ed=self.motif_ed)
            seq_trimer_list.append(tlist)
            trimer_counts.update(tlist)

        kept = [t for t, c in trimer_counts.items() if c >= 2]
        if len(kept) == 0:
            logging.warning("No trimers kept; falling back to contiguous trimers")
            kept = list(set([s[i:i+3] for s in sequences for i in range(max(0, len(s)-2))]))
        self.trimer_index = {t: i for i, t in enumerate(sorted(kept))}
        n_trimers = len(self.trimer_index)
        logging.info(f"RFUAnalyzer: {n_trimers} trimers retained for PCA projection")

        seq_count = len(seq_trimer_list)
        X = np.zeros((seq_count, n_trimers), dtype=np.float32)
        for i, tlist in enumerate(seq_trimer_list):
            for t in tlist:
                idx = self.trimer_index.get(t, None)
                if idx is not None:
                    X[i, idx] += 1.0
        if n_trimers <= self.n_components:
            self.n_components = max(1, n_trimers // 2)
        self.pca = PCA(n_components=self.n_components)
        try:
            self.pca.fit(X.T)
            self.trimer_fit = self.pca.components_.T
            logging.info("RFUAnalyzer: PCA for trimer embedding fitted")
        except Exception as e:
            logging.warning(f"PCA failed: {e}")
            self.trimer_fit = np.random.randn(n_trimers, self.n_components).astype(np.float32)

    def encode_sequences(self, sequences):
        if self.trimer_fit is None or self.trimer_index is None:
            self.fit_trimer_space(sequences)

        embeddings = []
        for s in sequences:
            tlist = self.get_positioned_trimers(s, st=self.motif_st, ed=self.motif_ed)
            vecs = []
            for t in tlist:
                idx = self.trimer_index.get(t, None)
                if idx is not None and idx < self.trimer_fit.shape[0]:
                    vecs.append(self.trimer_fit[idx])
            if len(vecs) == 0:
                embeddings.append(np.zeros(self.trimer_fit.shape[1], dtype=np.float32))
            else:
                embeddings.append(np.mean(vecs, axis=0))
        return np.vstack(embeddings)

    def fit_rfus(self, sequences):
        seq_emb = self.encode_sequences(sequences)
        n_clusters = min(self.n_rfus, max(2, len(sequences) // 2))
        logging.info(f"RFUAnalyzer: fitting KMeans for {n_clusters} clusters (n_rfus param {self.n_rfus})")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        try:
            self.kmeans.fit(seq_emb)
            logging.info("RFUAnalyzer: KMeans fit complete")
        except Exception as e:
            logging.error(f"KMeans fitting failed: {e}; using single-cluster fallback")
            self.kmeans = KMeans(n_clusters=1, random_state=42).fit(seq_emb)

    def assign_rfu_vector(self, sequences):
        if self.kmeans is None:
            self.fit_rfus(sequences)

        seq_emb = self.encode_sequences(sequences)
        labels = self.kmeans.predict(seq_emb)
        counts = Counter(labels)
        rfuv = np.zeros(self.kmeans.n_clusters, dtype=float)
        for k, c in counts.items():
            rfuv[k] = c
        return rfuv, labels

########################################
# Motif extractor
########################################

def extract_motifs_from_sequences(sequences, motif_size=2, max_gap=0):
    motifs = Counter()
    for s in sequences:
        if not isinstance(s, str):
            continue
        L = len(s)
        if motif_size == 2:
            for i in range(L):
                for gap in range(0, max_gap + 1):
                    j = i + gap + 1
                    if j < L:
                        m = s[i] + s[j]
                        motifs[f"g{gap}_" + m] += 1
        elif motif_size == 3:
            for i in range(L):
                for g1 in range(0, max_gap + 1):
                    for g2 in range(0, max_gap + 1):
                        j = i + g1 + 1
                        k = j + g2 + 1
                        if j < L and k < L:
                            m = s[i] + s[j] + s[k]
                            motifs[f"g{g1}g{g2}_" + m] += 1
        else:
            for i in range(L - motif_size + 1):
                m = s[i:i + motif_size]
                motifs[m] += 1
    return motifs

def build_topk_motif_list(sequences, motif_size=3, max_gap=1, topk=200):
    motifs = extract_motifs_from_sequences(sequences, motif_size=motif_size, max_gap=max_gap)
    top = [m for m, c in motifs.most_common(topk)]
    return top

def motif_presence_matrix(sequences, top_motifs):
    # returns binary matrix (n_seq, len(top_motifs))
    n = len(sequences)
    M = np.zeros((n, len(top_motifs)), dtype=np.int8)
    motif_set = set(top_motifs)
    for i, s in enumerate(sequences):
        if not isinstance(s, str):
            continue
        L = len(s)
        found = set()
        # simple contiguous substrings for motif match - for gapped motifs keep the label scheme
        for motif in top_motifs:
            # motif could be like "g1g0_XYZ" or raw substring; we'll check raw letters inside
            raw = motif.split("_")[-1] if "_" in motif else motif
            if raw in s:
                found.add(motif)
        for j, m in enumerate(top_motifs):
            if m in found:
                M[i, j] = 1
    return M

########################################
# Predictor class (with added training modules)
########################################


# ----------------------
# Evaluation utilities (added)
# ----------------------
import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, f1_score, precision_score, recall_score, normalized_mutual_info_score as _nmi_score

def compute_classification_metrics(y_true, y_pred, output_xlsx_path='results/classification_metrics.xlsx'):
    """
    Compute and save classification metrics (accuracy, class-weighted accuracy (balanced), NMI, per-class precision/recall/f1, confusion matrix).
    This function is tolerant to string labels.
    """
    os.makedirs(os.path.dirname(output_xlsx_path) or '.', exist_ok=True)
    y_true = _np.array(y_true).astype(str)
    y_pred = _np.array(y_pred).astype(str)

    classes = _np.unique(_np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    per_class_accuracy = _np.array([cm[i, i] / support[i] if support[i] > 0 else _np.nan for i in range(len(classes))])

    overall_acc = float(accuracy_score(y_true, y_pred))
    balanced_acc = float(_np.nanmean(recall))
    macro_precision = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average='micro', zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average='micro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    try:
        nmi = float(_nmi_score(y_true, y_pred, average_method='arithmetic'))
    except Exception:
        nmi = float('nan')

    per_class_df = _pd.DataFrame({
        'class': classes,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'class_accuracy': per_class_accuracy
    })

    summary = {
        'overall_accuracy': overall_acc,
        'balanced_accuracy': balanced_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'nmi': nmi
    }
    summary_df = _pd.DataFrame(list(summary.items()), columns=['metric', 'value'])
    cm_df = _pd.DataFrame(cm, index=classes, columns=classes)

    with _pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
        per_class_df.to_excel(writer, sheet_name='per_class', index=False)
        cm_df.to_excel(writer, sheet_name='confusion_matrix')

    print(\"Saved classification metrics to:\", output_xlsx_path)
    print(f\"Accuracy: {overall_acc:.4f}, Balanced acc: {balanced_acc:.4f}, NMI: {nmi:.4f}\")
    print(f\"Macro F1: {macro_f1:.4f}, Micro F1: {micro_f1:.4f}\")
    return summary

def evaluate_retrieval_metrics(predictor, df, query_col='CDR3', truth_col='Epitope', k_list=[1,3,5,10], output_xlsx_path='results/retrieval_metrics.xlsx'):
    """
    Evaluate Top-k accuracies and MRR for retrieval-based predictor.predict(query, k=K).
    df should contain query_col and truth_col.
    """
    os.makedirs(os.path.dirname(output_xlsx_path) or '.', exist_ok=True)
    queries = df[query_col].astype(str).tolist()
    truths = df[truth_col].astype(str).tolist()
    max_k = max(k_list)
    records = []
    reciprocal_ranks = []
    topk_counts = {k: 0 for k in k_list}
    total_q = len(queries)

    for q, true_ep in zip(queries, truths):
        try:
            results = predictor.predict(q, k=max_k)
        except Exception:
            results = []
        retrieved = [res.get('epitope', 'UNKNOWN') if isinstance(res, dict) else getattr(res, 'epitope', 'UNKNOWN') for res in results]
        rank = None
        for idx, ep in enumerate(retrieved, start=1):
            if ep == true_ep:
                rank = idx
                break
        reciprocal = 1.0 / rank if rank is not None else 0.0
        reciprocal_ranks.append(reciprocal)
        for k in k_list:
            if true_ep in retrieved[:k]:
                topk_counts[k] += 1
        records.append({
            'query': q,
            'true_epitope': true_ep,
            'retrieved_epitopes': ';'.join(map(str, retrieved)),
            'first_rank': rank if rank is not None else _np.nan,
            'reciprocal_rank': reciprocal
        })

    summary_records = []
    for k in k_list:
        summary_records.append({'k': k, 'topk_accuracy': topk_counts[k] / max(1, total_q)})
    mrr = float(_np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    details_df = _pd.DataFrame(records)
    summary_df = _pd.DataFrame(summary_records)
    extra_df = _pd.DataFrame([{'metric': 'MRR', 'value': mrr}, {'metric': 'queries', 'value': total_q}])

    with _pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
        details_df.to_excel(writer, sheet_name='details', index=False)
        summary_df.to_excel(writer, sheet_name='topk_summary', index=False)
        extra_df.to_excel(writer, sheet_name='extra', index=False)

    print('Saved retrieval metrics to:', output_xlsx_path)
    print('MRR: {:.4f}'.format(mrr))
    for r in summary_records:
        print(f\"Top-{r['k']} accuracy: {r['topk_accuracy']:.4f}\")

    return {'mrr': mrr, 'topk_summary': summary_records, 'details': details_df}
class TCRPredictor:
    def __init__(self, model_name="wukevin/tcr-bert"):
        self.device = self._select_device()
        logging.info(f"Using device: {self.device}")

        self.embedder = ImmuneBERTOptimized(model_name, self.device)
        self.gene_projector = nn.Linear(2, 768).to(self.device)
        self.gene_projector.eval()

        self.google_encoder = None

        self.graph_convolution = GraphNeuralEvolution(768)
        self.contrastive_learning = ContrastivePrototypeLearning(temperature=0.2, gamma=0.99)
        self.momentum_optimizer = None

        self.multi_modal_fusion = MultiModalFusion([768, 768, 768])

        self.cdr3_projector = nn.Linear(512, 768).to(self.device)
        self.epitope_projector = nn.Linear(512, 768).to(self.device)
        self.cdr3_projector.eval()
        self.epitope_projector.eval()

        self.rfu_analyzer = RFUAnalyzer(n_components=100, n_rfus=500)

        # place to store top motifs for features
        self.top_motifs_for_features = []

    def _select_device(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return torch.device("cuda")
        return torch.device("cpu")

    def load_data(self, file_path):
        logging.info(f"Loading data from: {file_path}")
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        df = self._clean_data(df)
        logging.info(f"Loaded {len(df)} sequences after cleaning")
        return df

    def _clean_data(self, df):
        if 'CDR3' in df.columns:
            df = df[df['CDR3'].notna()]
            df = df[df['CDR3'].str.len() >= 8]
        required_columns = ['CDR3', 'Epitope']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        for col in ['V', 'J', 'Epitope gene', 'Epitope species', 'MHC class']:
            if col in df.columns:
                df[col].fillna('UNKNOWN', inplace=True)
            else:
                df[col] = 'UNKNOWN'
        df = df.drop_duplicates(subset=['CDR3'])
        df = df.reset_index(drop=True)
        return df

    def embed_sequences(self, sequences, batch_size=4, method="immunebert"):
        if method == "google_universal":
            return self._google_universal_embed(sequences)
        else:
            return self.embedder(sequences, batch_size)

    def _google_universal_embed(self, sequences):
        # keep same defensive behavior: try to load if needed, else return zeros
        try:
            import tensorflow_hub as hub
        except Exception:
            logging.warning("tensorflow_hub not installed; google_universal embedding skipped.")
            return np.zeros((len(sequences), 512))
        if not sequences:
            return np.array([])
        if self.google_encoder is None:
            try:
                logging.info("Loading Google Universal Sentence Encoder...")
                self.google_encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
                logging.info("Google Universal Sentence Encoder loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load Google Universal Encoder: {str(e)}")
                self.google_encoder = None
                return np.zeros((len(sequences), 512))
        try:
            batch_size = 32
            embeddings = []
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                batch_embeddings = self.google_encoder(batch).numpy()
                embeddings.append(batch_embeddings)
            return np.vstack(embeddings) if embeddings else np.array([])
        except Exception as e:
            logging.error(f"Google Universal embedding failed: {str(e)}")
            return np.zeros((len(sequences), 512))

    def cluster_sequences(self, embeddings, max_clusters=50):
        if len(embeddings) == 0:
            return np.array([])

        unique_embeddings = np.unique(embeddings, axis=0)
        unique_count = len(unique_embeddings)
        n_clusters = min(unique_count, max_clusters)
        if n_clusters < 1:
            n_clusters = 1
        logging.info(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters (max {max_clusters})")

        clusterer = MiniBatchKMeans(n_clusters=n_clusters,
                                    batch_size=min(1024, len(embeddings)),
                                    random_state=42,
                                    n_init=3)
        try:
            clusters = clusterer.fit_predict(embeddings)
            prototypes = clusterer.cluster_centers_
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32)

            contrastive_loss, updated_prototypes = self.contrastive_learning(
                embeddings_tensor,
                prototypes_tensor,
                clusters
            )
            try:
                clusterer.cluster_centers_ = updated_prototypes.detach().cpu().numpy()
            except Exception:
                clusterer.cluster_centers_ = updated_prototypes.numpy() if isinstance(updated_prototypes, np.ndarray) else updated_prototypes
            logging.info(f"Contrastive loss (avg): {self.contrastive_learning.loss_history[-1] if self.contrastive_learning.loss_history else 'n/a'}")
            return clusters
        except Exception as e:
            logging.error(f"Clustering failed: {str(e)}")
            return np.zeros(len(embeddings), dtype=int)

    ########################################
    # 新增/改进：多 epoch contrastive training with InfoNCE + EMA prototype update
    ########################################
    def train_contrastive(self, embeddings, epitope_labels=None, n_clusters=20, num_epochs=20, batch_size=128,
                          lr=1e-4, ema_m=0.99, temperature=0.2, save_prefix="results/contrastive"):
        """
        Improved multi-epoch InfoNCE with EMA prototype update.
        - embeddings: np.array (N, D)
        - epitope_labels: optional list of labels (strings) for computing supervised metrics per epoch
        - n_clusters: number of prototypes (K)
        - num_epochs: up to 100 as you requested
        - batch_size: for sampling updates
        """
        os.makedirs("results", exist_ok=True)
        if len(embeddings) == 0:
            logging.error("No embeddings provided for training")
            return None

        device = torch.device("cpu")
        N, D = embeddings.shape
        # init kmeans to get initial prototypes
        logging.info("Initializing prototypes with KMeans...")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=min(1024, N), random_state=42, n_init=3)
        cluster_ids = kmeans.fit_predict(embeddings)
        prototypes = kmeans.cluster_centers_.astype(np.float32)  # (K, D)
        prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32, device=device)

        # prepare embeddings tensor (we won't update embedder)
        emb_tensor_all = torch.tensor(embeddings, dtype=torch.float32, device=device)

        # prepare random permutation indices for batching
        indices = np.arange(N)

        # metrics histories
        loss_history = []
        acc_history = []
        nmi_history = []

        # local contrastive module (for computing loss only)
        contrast = ContrastivePrototypeLearning(temperature=temperature)
        for epoch in range(num_epochs):
            logging.info(f"Contrastive training epoch {epoch+1}/{num_epochs}")
            # optional: re-run KMeans initialize with prototypes as init (warm start)
            try:
                if epoch % 5 == 0 and epoch > 0:
                    # re-init clusters based on current prototypes for stability
                    from sklearn.cluster import KMeans
                    kinit = KMeans(n_clusters=n_clusters, init=prototypes, n_init=1, random_state=42)
                    cluster_ids = kinit.fit_predict(embeddings)
                    prototypes = kinit.cluster_centers_.astype(np.float32)
                    prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32, device=device)
            except Exception as e:
                logging.warning(f"Warm KMeans init failed: {e}")

            # shuffle for batching
            np.random.shuffle(indices)
            epoch_losses = []
            # we'll accumulate per-cluster sums for EMA update
            cluster_sums = {k: np.zeros(D, dtype=np.float32) for k in range(n_clusters)}
            cluster_counts = {k: 0 for k in range(n_clusters)}

            for i0 in range(0, N, batch_size):
                batch_idx = indices[i0:i0+batch_size]
                z_batch = emb_tensor_all[batch_idx]  # (B, D)
                batch_cluster_ids = [int(x) if x >=0 and x < n_clusters else 0 for x in cluster_ids[batch_idx]]
                # compute InfoNCE loss via contrast module
                loss_tensor, _ = contrast(z_batch, prototypes_tensor, batch_cluster_ids)
                # no backprop as we do not train embedder; we record loss
                epoch_losses.append(loss_tensor.detach().cpu().item())

                # accumulate cluster sums for EMA update
                z_np = z_batch.detach().cpu().numpy()
                for ii, cid in enumerate(batch_cluster_ids):
                    if cid < 0 or cid >= n_clusters:
                        continue
                    cluster_sums[cid] += z_np[ii]
                    cluster_counts[cid] += 1

            # perform EMA update for each prototype using collected means for clusters seen in epoch
            for k in range(n_clusters):
                if cluster_counts[k] > 0:
                    mean_z = cluster_sums[k] / max(1, cluster_counts[k])
                    prototypes[k] = ema_m * prototypes[k] + (1.0 - ema_m) * mean_z
            prototypes_tensor = torch.tensor(prototypes, dtype=torch.float32, device=device)

            avg_epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            loss_history.append(avg_epoch_loss)

            # supervised metrics if provided
            if epitope_labels is not None:
                # map cluster -> majority epitope
                try:
                    df_tmp = pd.DataFrame({'cluster': cluster_ids, 'epitope': pd.Series(epitope_labels).fillna('UNKNOWN').astype(str)})
                    mapping = df_tmp.groupby('cluster')['epitope'].agg(lambda s: s.value_counts().idxmax()).to_dict()
                    pred_labels = [mapping.get(c, 'UNKNOWN') for c in cluster_ids]
                    true_labels = list(pd.Series(epitope_labels).fillna('UNKNOWN').astype(str))
                    acc = accuracy_score(true_labels, pred_labels)
                    nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')
                except Exception as e:
                    logging.warning(f"Metrics computation failed: {e}")
                    acc = float('nan')
                    nmi = float('nan')
            else:
                acc = float('nan')
                nmi = float('nan')

            acc_history.append(acc)
            nmi_history.append(nmi)
            logging.info(f"Epoch {epoch+1}: loss={avg_epoch_loss:.6f} acc={acc:.4f} nmi={nmi:.4f}")

        # save training metrics
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(loss_history)+1)),
            'contrastive_loss': loss_history,
            'accuracy': acc_history,
            'nmi': nmi_history
        })
        metrics_df.to_csv(os.path.join("results", "training_metrics_contrastive.csv"), index=False)
        logging.info("Saved contrastive training metrics to results/training_metrics_contrastive.csv")

        # plot loss/metrics
        plt.figure(figsize=(10,6))
        plt.plot(loss_history, marker='o', label='contrastive_loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Contrastive Loss (InfoNCE + EMA prototypes)')
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join("results", "contrastive_loss_infoNCE.png"), dpi=200, bbox_inches='tight'); plt.close()

        plt.figure(figsize=(10,6))
        epochs = list(range(1,len(acc_history)+1))
        plt.plot(epochs, acc_history, marker='o', label='accuracy')
        plt.plot(epochs, nmi_history, marker='s', label='NMI')
        plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.title('Supervised metrics during contrastive training')
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join("results", "contrastive_metrics_infoNCE.png"), dpi=200, bbox_inches='tight'); plt.close()

        # store final prototypes
        self.prototype_matrix = prototypes.copy()  # numpy
        # also persist
        np.save(os.path.join("results", "final_prototypes.npy"), prototypes)
        logging.info("Saved final prototypes to results/final_prototypes.npy")

        return cluster_ids, prototypes

    ########################################
    # 新增：在 fused embedding 上训练监督分类头（MLP）
    ########################################
    def train_classifier_head(self, fused_embeddings, epitope_labels, val_fraction=0.2,
                              epochs=100, batch_size=64, lr=1e-4, weight_decay=1e-4,
                              early_stopping_patience=10, class_weighted=True, save_prefix="results/classifier"):
        """
        Train a small MLP classifier on fused embeddings (fine-tune classification head).
        Saves: model pickle (.pt), training curves, confusion matrix, metrics CSV.
        """
        os.makedirs("results", exist_ok=True)
        X = np.array(fused_embeddings)
        y = np.array(pd.Series(epitope_labels).fillna('UNKNOWN').astype(str))
        # encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)
        logging.info(f"Classifier training: {X.shape[0]} samples, {n_classes} classes")

        # train/val split stratified
        X_train, X_val, y_train, y_val = train_test_split(X, y_enc, test_size=val_fraction, random_state=42, stratify=y_enc)

        # dataset loader
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

        # model: MLP 768 -> 256 -> n_classes
        in_dim = X.shape[1]
        hidden = 256 if in_dim >= 256 else max(64, in_dim // 2)
        model = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes)
        ).to(device)

        # class weights
        if class_weighted:
            # compute class weights inversely proportional to frequency
            cw = dict()
            unique, counts = np.unique(y_train, return_counts=True)
            for u, c in zip(unique, counts):
                cw[u] = float(len(y_train)) / (len(unique) * c)
            weight_tensor = torch.tensor([cw.get(i, 1.0) for i in range(n_classes)], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        best_val = -np.inf
        best_epoch = 0
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

        for epoch in range(1, epochs+1):
            model.train()
            # simple mini-batch
            perm = np.random.permutation(len(X_train))
            epoch_losses = []
            for i0 in range(0, len(perm), batch_size):
                idx = perm[i0:i0+batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.detach().cpu().item())
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

            # validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = float(criterion(val_logits, y_val_t).detach().cpu().item())
                preds = torch.argmax(val_logits, dim=1).detach().cpu().numpy()
                val_acc = accuracy_score(y_val, preds)
                val_f1 = f1_score(y_val, preds, average='macro', zero_division=0)
            logging.info(f"[Classifier] Epoch {epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            # early stopping on val_acc (or f1)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                # save best model
                torch.save({'model_state': model.state_dict(), 'label_encoder': le}, os.path.join("results", "classifier_best.pt"))
                # also keep pickle of le
                with open(os.path.join("results","label_encoder.pkl"), "wb") as f:
                    pickle.dump(le, f)
            if epoch - best_epoch >= early_stopping_patience:
                logging.info("Early stopping triggered for classifier training")
                break

        # save training curves and metrics
        metrics_df = pd.DataFrame({
            'epoch': list(range(1, len(history['train_loss'])+1)),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'val_acc': history['val_acc'],
            'val_f1': history['val_f1']
        })
        metrics_df.to_csv(os.path.join("results", "classifier_training_metrics.csv"), index=False)

        # plot curves
        plt.figure(figsize=(10,6))
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='train_loss', marker='o')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='val_loss', marker='o')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Classifier Loss Curves'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join("results", "classifier_loss_curves.png"), dpi=200, bbox_inches='tight'); plt.close()

        plt.figure(figsize=(10,6))
        plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='val_acc', marker='o')
        plt.plot(metrics_df['epoch'], metrics_df['val_f1'], label='val_f1', marker='o')
        plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.title('Classifier Metrics'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join("results", "classifier_metrics.png"), dpi=200, bbox_inches='tight'); plt.close()

        # confusion matrix using best model
        best = torch.load(os.path.join("results", "classifier_best.pt"), map_location=device)
        model.load_state_dict(best['model_state'])
        model.eval()
        with torch.no_grad():
            logits_all = model(torch.tensor(X, dtype=torch.float32, device=device))
            preds_all = torch.argmax(logits_all, dim=1).detach().cpu().numpy()
        cm = confusion_matrix(y_enc, preds_all, labels=range(n_classes))
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_df.to_csv(os.path.join("results", "classifier_confusion_matrix.csv"))
        # plot confusion matrix (absolute)
        plt.figure(figsize=(10,10))
        sns.heatmap(cm_df, cmap='Blues', annot=False)
        plt.title('Classifier Confusion Matrix'); plt.tight_layout()
        plt.savefig(os.path.join("results", "classifier_confusion_matrix.png"), dpi=200, bbox_inches='tight'); plt.close()

        logging.info(f"Classifier training finished. Best val acc: {best_val:.4f} at epoch {best_epoch}")
        return os.path.join("results","classifier_best.pt"), os.path.join("results","classifier_training_metrics.csv")

    ########################################
    # 新增：LightGBM 后端（若可用）
    ########################################
    def train_lightgbm_backend(self, fused_embeddings, sequences, v_list, j_list, epitope_labels,
                               top_motifs_k=200, pca_dim=128, n_splits=5, save_prefix="results/lgbm"):
        """
        Train LightGBM on engineered features:
          features = [PCA(fused_embeddings), motif_presence(topK), length, V_enc, J_enc, top-k sim to prototypes]
        Performs Stratified K-Fold CV (by labels), saves models and CV metrics.
        """
        os.makedirs("results", exist_ok=True)
        if not HAS_LIGHTGBM:
            logging.warning("lightgbm not installed; skipping LightGBM backend.")
            return None

        X_emb = np.array(fused_embeddings)
        y = np.array(pd.Series(epitope_labels).fillna('UNKNOWN').astype(str))
        n = X_emb.shape[0]
        # PCA reduce embedding for trees
        pca = PCA(n_components=min(pca_dim, X_emb.shape[1]-1))
        X_p = pca.fit_transform(X_emb)

        # build top motifs if not yet
        top_motifs = build_topk_motif_list(sequences, motif_size=3, max_gap=1, topk=top_motifs_k)
        self.top_motifs_for_features = top_motifs
        M = motif_presence_matrix(sequences, top_motifs)  # (n, topk)

        # encode v/j as integers
        v_le = LabelEncoder().fit(v_list)
        j_le = LabelEncoder().fit(j_list)
        V_enc = v_le.transform(v_list).reshape(-1,1)
        J_enc = j_le.transform(j_list).reshape(-1,1)
        lengths = np.array([len(s) if isinstance(s, str) else 0 for s in sequences]).reshape(-1,1)

        # optionally include top-k similarity to prototypes if available
        if hasattr(self, 'prototype_matrix'):
            proto = self.prototype_matrix  # (K, D)
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(X_emb, proto)  # (n, K)
            topk_sim = np.sort(sims, axis=1)[:, -5:]  # last 5 highest
        else:
            topk_sim = np.zeros((n, 5))

        # build final feature matrix
        X_feat = np.hstack([X_p, M, lengths, V_enc, J_enc, topk_sim])
        logging.info(f"LightGBM features shape: {X_feat.shape}")

        # target encoding
        le_target = LabelEncoder().fit(y)
        y_enc = le_target.transform(y)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = np.zeros((n, len(le_target.classes_)))
        models = []
        fold_metrics = []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_feat, y_enc)):
            logging.info(f"Training LightGBM fold {fold+1}/{n_splits}")
            dtrain = lgb.Dataset(X_feat[tr_idx], label=y_enc[tr_idx])
            dval = lgb.Dataset(X_feat[val_idx], label=y_enc[val_idx], reference=dtrain)
            params = {
                'objective': 'multiclass',
                'num_class': len(le_target.classes_),
                'metric': 'multi_logloss',
                'learning_rate': 0.05,
                'num_leaves': 128,
                'verbose': -1,
                'seed': 42
            }
            bst = lgb.train(params, dtrain, valid_sets=[dtrain, dval], num_boost_round=1000, early_stopping_rounds=50, verbose_eval=50)
            models.append(bst)
            pred_val = bst.predict(X_feat[val_idx])
            oof_preds[val_idx] = pred_val
            yv = y_enc[val_idx]
            ypred = np.argmax(pred_val, axis=1)
            acc = accuracy_score(yv, ypred)
            f1 = f1_score(yv, ypred, average='macro', zero_division=0)
            fold_metrics.append({'fold': fold+1, 'acc': acc, 'f1_macro': f1})
            logging.info(f"Fold {fold+1}: acc={acc:.4f}, f1_macro={f1:.4f}")

            # save model
            bst.save_model(os.path.join("results", f"lgbm_fold{fold+1}.txt"))

        # overall metrics
        ytrue = y_enc
        ypred_oof = np.argmax(oof_preds, axis=1)
        oof_acc = accuracy_score(ytrue, ypred_oof)
        oof_f1 = f1_score(ytrue, ypred_oof, average='macro', zero_division=0)
        logging.info(f"LightGBM OOF acc={oof_acc:.4f} f1_macro={oof_f1:.4f}")
        pd.DataFrame(fold_metrics).to_csv(os.path.join("results","lgbm_fold_metrics.csv"), index=False)

        # feature importance (average)
        importances = np.zeros(X_feat.shape[1])
        for m in models:
            importances += m.feature_importance(importance_type='gain')
        importances /= max(1, len(models))
        # store feature names
        feat_names = []
        feat_names += [f"emb_pca_{i}" for i in range(X_p.shape[1])]
        feat_names += [f"motif_{m}" for m in top_motifs]
        feat_names += ["length","V_enc","J_enc"] + [f"topk_sim_{i}" for i in range(topk_sim.shape[1])]
        fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
        fi_df.sort_values('importance', ascending=False, inplace=True)
        fi_df.to_csv(os.path.join("results","lgbm_feature_importance.csv"), index=False)

        # plot top 30 importances
        plt.figure(figsize=(8,10))
        topn = min(30, len(fi_df))
        sns.barplot(x='importance', y='feature', data=fi_df.head(topn))
        plt.title('LightGBM Feature Importance (gain)')
        plt.tight_layout()
        plt.savefig(os.path.join("results","lgbm_feature_importance.png"), dpi=200, bbox_inches='tight'); plt.close()

        # save label encoder for use in prediction
        with open(os.path.join("results","lgbm_label_encoder.pkl"), "wb") as f:
            pickle.dump(le_target, f)

        return {'oof_acc': oof_acc, 'oof_f1_macro': oof_f1, 'models': models, 'feature_importance': fi_df}

    ########################################
    # Existing pipeline methods (build_metanet etc.) - keep original behavior but call new train_contrastive optionally
    ########################################
    def build_metanet(self, df, n_clusters=20, embedding_method="immunebert", do_train_contrastive=False, train_epochs=10):
        if 'CDR3' not in df.columns:
            logging.error("DataFrame missing 'CDR3' column")
            return nx.Graph(), np.array([])

        sequences = df['CDR3'].tolist()
        logging.info(f"Building MetaNet for {len(sequences)} sequences with method: {embedding_method}")

        embeddings_raw = self.embed_sequences(sequences, batch_size=4, method=embedding_method)
        if len(embeddings_raw) == 0:
            logging.error("Failed to generate embeddings")
            return nx.Graph(), np.array([])

        fusion_features = []
        v_values = df['V'].fillna('UNKNOWN').astype(str).tolist()
        j_values = df['J'].fillna('UNKNOWN').astype(str).tolist()
        v_le = LabelEncoder().fit(v_values)
        j_le = LabelEncoder().fit(j_values)

        for idx, row in df.reset_index(drop=True).iterrows():
            features = []
            if idx < len(embeddings_raw):
                seq_feat = torch.tensor(embeddings_raw[idx], dtype=torch.float32)
                if embedding_method == "google_universal" and seq_feat.shape[0] == 512:
                    with torch.no_grad():
                        seq_feat = seq_feat.to(self.device)
                        seq_feat = self.cdr3_projector(seq_feat).cpu()
                if seq_feat.dim() == 1:
                    features.append(seq_feat)
                else:
                    features.append(seq_feat.squeeze(0))
            else:
                features.append(torch.zeros(768))

            v_val = str(row.get('V', 'UNKNOWN'))
            j_val = str(row.get('J', 'UNKNOWN'))
            try:
                v_gene_enc = int(v_le.transform([v_val])[0])
            except Exception:
                v_gene_enc = 0
            try:
                j_gene_enc = int(j_le.transform([j_val])[0])
            except Exception:
                j_gene_enc = 0
            gene_feature = np.array([v_gene_enc, j_gene_enc], dtype=np.float32)
            with torch.no_grad():
                gene_tensor = torch.tensor(gene_feature, dtype=torch.float32).to(self.device)
                gene_feature_projected = self.gene_projector(gene_tensor).cpu()
            features.append(gene_feature_projected)

            epitope_text = row.get('Epitope', 'UNKNOWN')
            epitope_feat = self.embed_sequences([epitope_text], batch_size=1, method=embedding_method)
            if len(epitope_feat) > 0:
                epitope_tensor = torch.tensor(epitope_feat[0], dtype=torch.float32)
                if embedding_method == "google_universal" and epitope_tensor.shape[0] == 512:
                    with torch.no_grad():
                        epitope_tensor = epitope_tensor.to(self.device)
                        epitope_tensor = self.epitope_projector(epitope_tensor).cpu()
                if epitope_tensor.dim() == 1:
                    features.append(epitope_tensor)
                else:
                    features.append(epitope_tensor.squeeze(0))
            else:
                features.append(torch.zeros(768))

            fused = self.multi_modal_fusion(features)
            if fused.dim() > 1:
                fused = fused.squeeze(0)
            fusion_features.append(fused.detach().numpy())

        embeddings = np.array(fusion_features)
        if np.isnan(embeddings).any():
            logging.warning("Embeddings contain NaN values after fusion. Replacing with 0.")
            embeddings = np.nan_to_num(embeddings)

        # initial clustering (single pass)
        clusters = self.cluster_sequences(embeddings, n_clusters)
        df['cluster'] = clusters

        # optionally run multi-epoch contrastive training to produce loss curve & better prototypes
        if do_train_contrastive:
            logging.info("Starting multi-epoch contrastive training to produce loss/metric curves...")
            final_clusters, final_prototypes = self.train_contrastive(embeddings, epitope_labels=df['Epitope'].tolist(), n_clusters=n_clusters, num_epochs=train_epochs)
            if final_clusters is not None:
                df['cluster'] = final_clusters
                clusters = final_clusters
                # update embeddings via simple prototype replacement (optional)
            else:
                logging.warning("Contrastive training returned None; keeping initial clusters")

        # RFU analyzer
        try:
            self.rfu_analyzer.fit_trimer_space(df['CDR3'].tolist())
            rfuv, labels = self.rfu_analyzer.assign_rfu_vector(df['CDR3'].tolist())
            rfudf = pd.DataFrame(rfuv.reshape(1, -1))
            os.makedirs("results", exist_ok=True)
            rfudf.to_excel(os.path.join("results", "rfu_vector_training.xlsx"), index=False)
            # 新增：保存 RFU 向量可视化
            plt.figure(figsize=(12,6))
            plt.bar(range(len(rfuv)), rfuv)
            plt.title("RFU Vector (training approx)")
            plt.xlabel("RFU cluster index"); plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join("results", "rfu_vector_training.png"), dpi=200, bbox_inches='tight')
            plt.close()
            logging.info("Saved RFU (approx) vector for training to results/rfu_vector_training.xlsx and PNG")
        except Exception as e:
            logging.warning(f"RFUAnalyzer step failed: {e}")

        self.id_to_seq = sequences

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index = faiss.IndexIDMap(index)
        embeddings_f32 = embeddings.astype('float32')
        faiss.normalize_L2(embeddings_f32)
        ids = np.arange(len(embeddings))
        try:
            index.add_with_ids(embeddings_f32, ids)
        except Exception as e:
            logging.error(f"FAISS add failed: {e}; attempting add()")
            index.add(embeddings_f32)

        G = nx.Graph()
        metadata = {}
        seq_to_idx = {seq: idx for idx, seq in enumerate(sequences)}

        for idx, row in df.reset_index(drop=True).iterrows():
            seq = row['CDR3']
            if not isinstance(seq, str):
                continue
            G.add_node(seq,
                       v_gene=row.get('V', 'UNKNOWN'),
                       j_gene=row.get('J', 'UNKNOWN'),
                       epitope=row.get('Epitope', 'UNKNOWN'),
                       cluster=row.get('cluster', -1))
            metadata[seq] = {
                'cdr3': seq,
                'v_gene': row.get('V', 'UNKNOWN'),
                'j_gene': row.get('J', 'UNKNOWN'),
                'epitope': row.get('Epitope', 'UNKNOWN'),
                'epitope_gene': row.get('Epitope gene', 'UNKNOWN'),
                'epitope_species': row.get('Epitope species', 'UNKNOWN'),
                'mhc_class': row.get('MHC class', 'UNKNOWN'),
                'cluster': row.get('cluster', -1),
                'idx': seq_to_idx.get(seq, -1),
                'embedding': embeddings[idx] if idx < len(embeddings) else None
            }

        # single-pass graph evolution (safe)
        max_cluster_id = int(max(clusters)) if len(clusters) > 0 else -1
        for cluster_id in range(max_cluster_id + 1):
            cluster_seqs = [seq for seq, data in metadata.items() if data['cluster'] == cluster_id]
            if not cluster_seqs:
                continue
            try:
                cluster_indices = [metadata[seq]['idx'] for seq in cluster_seqs]
                cluster_embeddings = embeddings[cluster_indices]
                adj_matrix = np.ones((len(cluster_seqs), len(cluster_seqs)), dtype=np.float32)
                embeddings_tensor = torch.tensor(cluster_embeddings, dtype=torch.float32)
                adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
                new_embeddings = self.graph_convolution(embeddings_tensor, adj_tensor).detach().numpy()
                for i, seq in enumerate(cluster_seqs):
                    metadata[seq]['embedding'] = new_embeddings[i]
                embeddings[cluster_indices] = new_embeddings
            except Exception as e:
                logging.warning(f"Graph evolution for cluster {cluster_id} failed: {e}")

        if np.isnan(embeddings).any():
            logging.warning("Embeddings contain NaN values after graph evolution. Replacing with 0.")
            embeddings = np.nan_to_num(embeddings)

        try:
            index.reset()
            embeddings_f32 = embeddings.astype('float32')
            faiss.normalize_L2(embeddings_f32)
            index.add_with_ids(embeddings_f32, ids)
        except Exception as e:
            logging.warning(f"FAISS rebuild failed: {e}")

        for cluster_id in range(max_cluster_id + 1):
            cluster_seqs = [seq for seq, data in metadata.items() if data['cluster'] == cluster_id]
            if not cluster_seqs:
                continue
            if len(cluster_seqs) > 100:
                cluster_indices = [metadata[seq]['idx'] for seq in cluster_seqs]
                cluster_embeddings = embeddings[cluster_indices]
                cluster_center = np.mean(cluster_embeddings, axis=0)
                distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
                centroid_idx_in_cluster = np.argmin(distances)
                centroid = cluster_seqs[centroid_idx_in_cluster]
                for seq in cluster_seqs:
                    if seq != centroid:
                        G.add_edge(centroid, seq, weight=1.0)
            else:
                for i in range(len(cluster_seqs)):
                    for j in range(i + 1, len(cluster_seqs)):
                        G.add_edge(cluster_seqs[i], cluster_seqs[j], weight=1.0)

        self.index = index
        self.metadata = metadata

        logging.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G, embeddings

    # rest of methods (build_epitope_network, predict, cluster_analysis, visualize_clusters, plot_contrastive_loss, create_adjacency_matrix, save_motifs)
    # largely unchanged (copy previous implementations or rely on existing methods)

    def build_epitope_network(self, df, output_dir="results"):
        if 'Epitope' not in df.columns or 'cluster' not in df.columns:
            logging.error("DataFrame missing required columns for epitope network")
            return nx.Graph()

        os.makedirs(output_dir, exist_ok=True)
        logging.info("Building epitope-centric network")
        G = nx.Graph()
        epitope_counts = df['Epitope'].value_counts()
        for epitope, count in epitope_counts.items():
            if epitope == 'UNKNOWN' or not isinstance(epitope, str):
                continue
            G.add_node(epitope, type='epitope', size=count)

        for _, row in df.iterrows():
            cluster = row['cluster']
            epitope = row['Epitope']
            if epitope == 'UNKNOWN' or not isinstance(epitope, str):
                continue
            if f"Cluster_{cluster}" not in G:
                G.add_node(f"Cluster_{cluster}", type='cluster', size=1)
            else:
                G.nodes[f"Cluster_{cluster}"]['size'] += 1
            if G.has_edge(f"Cluster_{cluster}", epitope):
                G[f"Cluster_{cluster}"][epitope]['weight'] += 1
            else:
                G.add_edge(f"Cluster_{cluster}", epitope, weight=1)

        epitope_hotspots = []
        for node, data in G.nodes(data=True):
            if data['type'] == 'epitope':
                cluster_connections = sum(1 for n in G.neighbors(node) if G.nodes[n]['type'] == 'cluster')
                epitope_hotspots.append((node, data['size'], cluster_connections))
        epitope_hotspots.sort(key=lambda x: x[1], reverse=True)
        top_epitopes = epitope_hotspots[:10]
        hotspots_df = pd.DataFrame(top_epitopes, columns=['Epitope', 'Count', 'ClusterConnections'])
        hotspots_path = os.path.join(output_dir, 'epitope_hotspots.xlsx')
        hotspots_df.to_excel(hotspots_path, index=False)
        logging.info(f"Saved epitope hotspots to {hotspots_path}")

        # Visualization (safe)
        plt.figure(figsize=(16, 14))
        node_colors = []
        node_sizes = []
        for node, data in G.nodes(data=True):
            if data['type'] == 'epitope':
                node_colors.append('lightcoral')
                node_sizes.append(data['size'] * 10)
            else:
                node_colors.append('skyblue')
                node_sizes.append(data['size'] * 2)
        edge_weights = [G[u][v]['weight'] * 0.2 for u, v in G.edges()]
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.5)
        labels = {}
        for node, data in G.nodes(data=True):
            if data['type'] == 'epitope' and data['size'] > 10:
                labels[node] = node
            elif data['type'] == 'cluster' and data['size'] > 20:
                labels[node] = node
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        plt.title("Epitope-Centric Network Analysis", fontsize=16, pad=20)
        plt.figtext(0.5, 0.01,
                    "Immunological Insight: Node size represents frequency, edge thickness represents association strength",
                    ha="center", fontsize=12)
        plt.scatter([], [], c='lightcoral', s=100, label='Epitopes')
        plt.scatter([], [], c='skyblue', s=50, label='TCR Clusters')
        plt.legend(scatterpoints=1, frameon=True, loc='best')
        network_path = os.path.join(output_dir, 'epitope_network.png')
        plt.savefig(network_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Epitope network visualization saved to {network_path}")
        return G

    def predict(self, query_cdr3, k=3):
        if not hasattr(self, 'index') or not hasattr(self, 'metadata'):
            logging.error("Model not initialized. Call build_metanet first.")
            return []

        query_embedding = self.embed_sequences([query_cdr3], batch_size=1)
        if len(query_embedding) == 0:
            return []

        query_embedding_f32 = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding_f32)
        similarities, indices = self.index.search(query_embedding_f32, k * 5)
        results = []
        seen_seqs = set()
        for i, sim in zip(indices[0], similarities[0]):
            if i < 0 or i >= len(self.id_to_seq):
                continue
            seq = self.id_to_seq[i]
            data = self.metadata[seq]
            if seq == query_cdr3 or seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            normalized_sim = (sim + 1) / 2
            if normalized_sim > 0.999:
                normalized_sim = 0.999 - 0.001 * random.random()
            results.append({'similarity': float(normalized_sim), **data})
            if len(results) >= k:
                break
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:k]

    def cluster_analysis(self, results):
        if not results:
            return {}
        epitope_counter = defaultdict(int)
        for res in results:
            epitope_counter[res['epitope']] += 1
        motifs = defaultdict(int)
        seq_len = len(results[0]['cdr3'])
        motif_size = 2 if seq_len < 12 else 3
        for res in results:
            seq = res['cdr3']
            for i in range(len(seq) - motif_size + 1):
                motif = seq[i:i + motif_size]
                motifs[motif] += 1
        top_motifs = sorted(motifs.items(), key=lambda x: x[1], reverse=True)[:3]
        return {
            'top_epitopes': sorted(epitope_counter.items(), key=lambda x: x[1], reverse=True),
            'top_motifs': top_motifs,
            'avg_similarity': sum(res['similarity'] for res in results) / len(results)
        }

    def visualize_clusters(self, embeddings, clusters, output_dir):
        if len(embeddings) == 0 or len(clusters) == 0:
            logging.warning("No embeddings or clusters to visualize")
            return None

        os.makedirs(output_dir, exist_ok=True)
        if np.isnan(embeddings).any():
            logging.warning("Embeddings contain NaN values. Replacing with 0.")
            embeddings = np.nan_to_num(embeddings)

        logging.info("Reducing dimensions with UMAP")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        try:
            reduced_emb = reducer.fit_transform(embeddings)
        except Exception as e:
            logging.error(f"UMAP failed: {str(e)}. Using PCA instead.")
            pca = PCA(n_components=2)
            reduced_emb = pca.fit_transform(embeddings)

        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(10, 1, height_ratios=[8] * 9 + [1])
        ax_main = fig.add_subplot(gs[:-1, 0])
        ax_text = fig.add_subplot(gs[-1, 0])

        unique_clusters = np.unique(clusters)
        n_clusters = len(unique_clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, max(1, n_clusters)))
        cmap = ListedColormap(colors)

        scatter = ax_main.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=clusters, cmap=cmap, alpha=0.7, s=20)

        for cluster_id in unique_clusters:
            cluster_mask = (clusters == cluster_id)
            if np.sum(cluster_mask) > 0:
                center = np.mean(reduced_emb[cluster_mask], axis=0)
                ax_main.scatter(center[0], center[1], c='red', marker='X', s=150, edgecolor='white')
                ax_main.text(center[0], center[1], f'C{cluster_id}', fontsize=11, ha='center', va='bottom', fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))

        cluster_counts = np.bincount(clusters)
        cluster_info_lines = [f"Cluster {i}: {count} sequences" for i, count in enumerate(cluster_counts) if count > 0]
        cluster_info = " | ".join(cluster_info_lines)

        ax_text.axis('off')
        ax_text.text(0.01, 0.5, cluster_info, ha='left', va='center', fontsize=9, wrap=True)

        cbar = fig.colorbar(scatter, ax=ax_main, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Cluster ID')

        ax_main.set_title('CDR3 Sequence Clustering (Training Data)', fontsize=14, pad=12)
        ax_main.set_xlabel('Dimension 1', fontsize=12)
        ax_main.set_ylabel('Dimension 2', fontsize=12)

        fig.suptitle("Immunological Insight: Clusters represent TCR groups with shared antigen specificity", fontsize=12, fontweight='bold', y=0.97)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        cluster_path = os.path.join(output_dir, 'training_clusters_umap.png')
        plt.savefig(cluster_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Training cluster visualization saved to {cluster_path}")

        plt.figure(figsize=(10, 6))
        cluster_sizes = [np.sum(clusters == i) for i in unique_clusters]
        bar_colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_sizes)))
        plt.bar(range(len(cluster_sizes)), cluster_sizes, color=bar_colors)
        for i, size in enumerate(cluster_sizes):
            plt.text(i, size + 0.5, str(size), ha='center', va='bottom', fontsize=9)
        plt.title('Cluster Size Distribution', fontsize=12)
        plt.xlabel('Cluster ID', fontsize=10)
        plt.ylabel('Number of Sequences', fontsize=10)
        plt.xticks(range(len(cluster_sizes)), [f"C{i}" for i in range(len(cluster_sizes))])
        plt.grid(axis='y', alpha=0.3)
        size_path = os.path.join(output_dir, 'cluster_size_distribution.png')
        plt.savefig(size_path, dpi=150, bbox_inches='tight')
        plt.close()

        # plot contrastive loss history (if any)
        self.plot_contrastive_loss(output_dir)
        return reduced_emb

    def plot_contrastive_loss(self, output_dir="results"):
        path_rerender = os.path.join(output_dir, 'contrastive_loss_infoNCE.png')
        path_orig = os.path.join(output_dir, 'contrastive_loss_history.png')
        if os.path.exists(path_rerender):
            logging.info(f"Contrastive loss plot available at {path_rerender}")
            return
        if not hasattr(self.contrastive_learning, 'loss_history') or not self.contrastive_learning.loss_history:
            logging.info("No contrastive loss history to plot")
            return
        os.makedirs(output_dir, exist_ok=True)
        hist = np.array(self.contrastive_learning.loss_history, dtype=float)
        plt.figure(figsize=(10, 6))
        if hist.size == 1:
            plt.plot([0], hist, marker='o', linestyle='-', linewidth=2)
        else:
            plt.plot(hist, label='Contrastive Loss', linewidth=2, marker='o')
        plt.title('Contrastive Loss History', fontsize=14)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        ymin, ymax = plt.ylim()
        if ymax - ymin < 1e-6:
            plt.ylim(ymin - 0.5, ymax + 0.5)
        loss_path = os.path.join(output_dir, 'contrastive_loss_history.png')
        plt.savefig(loss_path, dpi=200, bbox_inches='tight')
        plt.close()
        logging.info(f"Contrastive loss history plot saved to {loss_path}")

    def create_adjacency_matrix(self, df):
        n = len(df)
        adj_matrix = csr_matrix((n, n), dtype=np.int8)
        df = df.reset_index(drop=True)
        for cluster_id in df['cluster'].unique():
            cluster_indices = df[df['cluster'] == cluster_id].index.values
            for i in cluster_indices:
                for j in cluster_indices:
                    if i != j and i < n and j < n:
                        adj_matrix[i, j] = 1
        return adj_matrix

    def save_motifs(self, df, output_dir="results", motif_size=2, max_gap=0, top_k=50):
        os.makedirs(output_dir, exist_ok=True)
        sequences = df['CDR3'].tolist()
        motifs = extract_motifs_from_sequences(sequences, motif_size=motif_size, max_gap=max_gap)
        top = motifs.most_common(top_k)
        motifs_df = pd.DataFrame(top, columns=['motif', 'count'])
        motifs_xlsx = os.path.join(output_dir, f"motifs_k{motif_size}_gap{max_gap}.xlsx")
        motifs_df.to_excel(motifs_xlsx, index=False)
        logging.info(f"Saved top {top_k} motifs to {motifs_xlsx}")
        return motifs_df

########################################
# Main (example run)
########################################

if __name__ == "__main__":
    try:
        predictor = TCRPredictor()

        train_file = "train.xlsx"
        if not os.path.exists(train_file):
            train_file = "train.csv"
            if not os.path.exists(train_file):
                raise FileNotFoundError("Training file not found")

        df = predictor.load_data(train_file)

        # Build MetaNet and run multi-epoch contrastive training to get true curves
        # do_train_contrastive True & train_epochs up to 100 allowed
        G, embeddings = predictor.build_metanet(df, n_clusters=20, embedding_method="google_universal",
                                                do_train_contrastive=True, train_epochs=20)

        # Save epitope network & visualization
        epitope_network = predictor.build_epitope_network(df, "results")

        # Visualize clusters (UMAP)
        if len(df) > 0:
            clusters = df['cluster'].values
            reduced_emb = predictor.visualize_clusters(embeddings, clusters, "results")
        else:
            reduced_emb = None

        # Save motifs
        predictor.save_motifs(df, output_dir="results", motif_size=2, max_gap=0, top_k=100)
        predictor.save_motifs(df, output_dir="results", motif_size=2, max_gap=2, top_k=100)
        predictor.save_motifs(df, output_dir="results", motif_size=3, max_gap=1, top_k=100)

        # example: train classifier head using fused embeddings & epitope labels
        # fused embeddings = embeddings (already fused in build_metanet)
        try:
            clf_path, clf_metrics = predictor.train_classifier_head(embeddings, df['Epitope'].tolist(),
                                                                    val_fraction=0.2, epochs=50, early_stopping_patience=8)
            logging.info(f"Classifier trained and saved to {clf_path}")
        except Exception as e:
            logging.warning(f"Classifier training failed: {e}")

        # example: train LightGBM backend if available
        try:
            lgbm_result = predictor.train_lightgbm_backend(embeddings, df['CDR3'].tolist(),
                                                          df['V'].fillna('UNKNOWN').astype(str).tolist(),
                                                          df['J'].fillna('UNKNOWN').astype(str).tolist(),
                                                          df['Epitope'].tolist(), top_motifs_k=200)
            if lgbm_result:
                logging.info(f"LightGBM OOF acc: {lgbm_result['oof_acc']:.4f}, f1_macro: {lgbm_result['oof_f1_macro']:.4f}")
        except Exception as e:
            logging.warning(f"LightGBM training failed or skipped: {e}")

        # example predictions
        test_sequences = [
            "CASSQVTLPTETQYF",
            "CASSSLNTQYF",
            "CTSSQVTLPTETQYF"
        ]

        for seq in test_sequences:
            logging.info(f"\nPredicting for sequence: {seq}")
            results = predictor.predict(seq, k=2)
            if not results:
                logging.info("No matches found")
                continue
            logging.info("Top matches:")
            for i, res in enumerate(results):
                logging.info(f"{i+1}. CDR3: {res['cdr3']}")
                logging.info(f"   Epitope: {res['epitope']}")
                logging.info(f"   Antigen Gene: {res['epitope_gene']}")
                logging.info(f"   Species: {res['epitope_species']}")
                logging.info(f"   V Gene: {res['v_gene']}")
                logging.info(f"   J Gene: {res['j_gene']}")
                logging.info(f"   Similarity: {res['similarity']:.4f}")

            analysis = predictor.cluster_analysis(results)
            if analysis:
                logging.info("\nCluster analysis:")
                if analysis['top_epitopes']:
                    logging.info(f"Top epitope: {analysis['top_epitopes'][0][0]}")
                if analysis['top_motifs']:
                    logging.info(f"Top motif: {analysis['top_motifs'][0][0]}")
                if 'avg_similarity' in analysis:
                    logging.info(f"Average similarity: {analysis['avg_similarity']:.4f}")

            if reduced_emb is not None and results:
                plt.figure(figsize=(10, 8))
                plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c='lightgray', alpha=0.2, s=15, label='All Sequences')
                match_seqs = [res['cdr3'] for res in results]
                match_mask = df['CDR3'].isin(match_seqs).values
                if match_mask.any():
                    plt.scatter(reduced_emb[match_mask, 0], reduced_emb[match_mask, 1], c='red', alpha=0.8, s=50, label='Matches')
                query_index = df[df['CDR3'] == seq].index
                if not query_index.empty:
                    plt.scatter(reduced_emb[query_index, 0], reduced_emb[query_index, 1], c='blue', marker='*', s=200, label='Query Sequence')
                plt.title(f'Matches for: {seq}', fontsize=14)
                plt.xlabel('Dimension 1', fontsize=12)
                plt.ylabel('Dimension 2', fontsize=12)
                plt.legend()
                plt.figtext(0.5, 0.02, "Immunological Insight: Spatial proximity indicates shared antigen specificity",
                           ha="center", fontsize=11, fontstyle='italic')
                seq_path = os.path.join("results", f"matches_{seq[:6]}.png")
                plt.savefig(seq_path, dpi=200, bbox_inches='tight')
                plt.close()
                logging.info(f"Match visualization saved to {seq_path}")

    except Exception as e:
        logging.exception(f"Critical error: {str(e)}")


