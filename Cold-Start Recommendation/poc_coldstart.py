# -*- coding: utf-8 -*-
"""
콜드 스타트 추천시스템: 불확실성 인식 대조학습
==============================================

콜드 스타트 문제를 해결하기 위한 추천시스템 구현
- 사용자-아이템 대조학습 (U-I Contrastive Learning)
- 표현 향상 대조학습 (R-E Contrastive Learning)
- 가우시안 임베딩을 통한 불확실성 추정
- 콜드/웜 아이템 분할 평가
"""

import os
import json
import gzip
import math
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp

# Google Colab 환경 설정
from google.colab import drive
try:
    drive.mount('/content/Drive')
except:
    pass

# =============================================================================
# 설정 및 하이퍼파라미터
# =============================================================================

# 재현성을 위한 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 데이터 경로 설정
DATA_DIR = "/content/Drive/MyDrive/kakao/cold_start/data/"
REVIEWS_PATH = os.path.join(DATA_DIR, "Magazine_Subscriptions.json")
META_PATH = os.path.join(DATA_DIR, "meta_Magazine_Subscriptions.json")

# 텍스트 임베딩 설정
SBERT_MODEL = "all-MiniLM-L6-v2"  # 384차원 임베딩
MAX_REVIEWS_PER_ITEM = 50
MAX_TEXT_CHARS = 800
TEXT_BATCH = 64

# 데이터 분할 설정
COLD_ITEM_RATIO = 0.2              # 20% 아이템을 콜드 아이템으로 설정
WARM_SPLIT = (0.8, 0.1, 0.1)       # 웜 아이템의 train/val/test 비율

# 학습 하이퍼파라미터
NEG_PER_POS = 50                   # 양성 샘플당 음성 샘플 수
TAU_UI = 0.07                      # U-I 대조학습 온도
TAU_RE = 0.07                      # R-E 대조학습 온도
LAMBDA_RE = 0.4                    # R-E 손실 가중치: (1-λ)*L_UI + λ*L_RE
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
EVAL_EVERY = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 실행 환경 정보 출력
print("🚀 실행 환경 정보:")
print(f"   • Device: {device}")
if torch.cuda.is_available():
    print(f"   • GPU: {torch.cuda.get_device_name(0)}")
    print(f"   • GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"   • PyTorch Version: {torch.__version__}")
print()

# =============================================================================
# 유틸리티 함수
# =============================================================================

def set_seed(seed=SEED):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================================================================
# 평가 메트릭
# =============================================================================

def recall_at_k(ranked_items, positive_items, k=10):
    """
    Recall@K 계산

    Args:
        ranked_items: 순위가 매겨진 아이템 인덱스 리스트
        positive_items: 실제 상호작용이 있는 아이템 집합
        k: 상위 K개 아이템
    """
    topk = ranked_items[:k]
    hits = sum(1 for item in topk if item in positive_items)
    denominator = min(k, len(positive_items)) if len(positive_items) > 0 else 1
    return hits / denominator

def ndcg_at_k(ranked_items, positive_items, k=10):
    """
    NDCG@K 계산 (이진 관련성)

    Args:
        ranked_items: 순위가 매겨진 아이템 인덱스 리스트
        positive_items: 실제 상호작용이 있는 아이템 집합
        k: 상위 K개 아이템
    """
    dcg = 0.0
    for rank, item in enumerate(ranked_items[:k], start=1):
        if item in positive_items:
            dcg += 1.0 / math.log2(rank + 1)

    # 이상적인 DCG 계산
    ideal_hits = min(k, len(positive_items))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0

# =============================================================================
# 데이터 로딩 및 전처리
# =============================================================================

def _open_jsonl(path):
    """JSONL 파일 열기 (압축/비압축 지원)"""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")

def load_reviews_jsonl(path, keep_fields=("reviewerID", "asin", "summary", "reviewText")):
    """
    JSONL 파일에서 리뷰 데이터 로딩

    Returns:
        ui_pairs: (사용자_id, 아이템_id) 튜플 리스트
        texts_by_item: 아이템_id별 리뷰 텍스트 리스트 매핑
    """
    ui_pairs = []
    texts_by_item = defaultdict(list)

    with _open_jsonl(path) as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue

            user_id = data.get("reviewerID")
            item_id = data.get("asin")
            if not user_id or not item_id:
                continue

            ui_pairs.append((user_id, item_id))

            # 요약과 리뷰 텍스트 결합
            summary = data.get("summary") or ""
            review_text = data.get("reviewText") or ""
            combined_text = (summary + " . " + review_text).strip()
            if combined_text:
                texts_by_item[item_id].append(combined_text)

    return ui_pairs, texts_by_item

def load_meta_jsonl(path):
    """
    JSONL 파일에서 아이템 메타데이터 로딩

    Returns:
        item_to_meta: 아이템_id별 메타데이터 텍스트 매핑
    """
    item_to_meta = {}

    with _open_jsonl(path) as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue

            item_id = data.get("asin")
            if not item_id:
                continue

            title = str(data.get("title") or "")
            categories = data.get("category") or data.get("categories")

            # 중첩된 카테고리 리스트 평면화
            if isinstance(categories, list):
                flat_cats = []
                def flatten(x):
                    for item in x:
                        if isinstance(item, list):
                            flatten(item)
                        else:
                            flat_cats.append(str(item))
                flatten(categories)
                cat_text = " . ".join(flat_cats)
            elif categories:
                cat_text = str(categories)
            else:
                cat_text = ""

            meta_text = (title + " . " + cat_text).strip()
            item_to_meta[item_id] = meta_text

    return item_to_meta

def create_data_splits(ui_pairs, cold_item_ratio=COLD_ITEM_RATIO, warm_split=WARM_SPLIT):
    """
    데이터를 콜드/웜 아이템과 train/val/test 세트로 분할

    Returns:
        train_pairs, val_warm_pairs, val_cold_pairs,
        test_warm_pairs, test_cold_pairs와 ID 매핑을 포함한 딕셔너리
    """
    # ID 매핑 생성
    users = sorted({u for u, _ in ui_pairs})
    items = sorted({i for _, i in ui_pairs})
    user_to_id = {u: i for i, u in enumerate(users)}
    item_to_id = {i: idx for idx, i in enumerate(items)}

    # 아이템을 콜드와 웜으로 분할
    set_seed(SEED)
    all_items = list(items)
    cold_count = max(1, int(cold_item_ratio * len(all_items)))
    cold_items = set(random.sample(all_items, cold_count))
    warm_items = set(all_items) - cold_items

    # 상호작용 분할
    train_pairs = []
    val_warm_pairs = []
    val_cold_pairs = []
    test_warm_pairs = []
    test_cold_pairs = []

    p_train, p_val, p_test = warm_split

    for user, item in ui_pairs:
        user_idx = user_to_id[user]
        item_idx = item_to_id[item]

        if item in cold_items:
            # 콜드 아이템: val/test만 (학습 데이터 없음)
            if random.random() < 0.5:
                val_cold_pairs.append((user_idx, item_idx))
            else:
                test_cold_pairs.append((user_idx, item_idx))
        else:
            # 웜 아이템: train/val/test
            r = random.random()
            if r < p_train:
                train_pairs.append((user_idx, item_idx))
            elif r < p_train + p_val:
                val_warm_pairs.append((user_idx, item_idx))
            else:
                test_warm_pairs.append((user_idx, item_idx))

    return {
        'train_pairs': train_pairs,
        'val_warm_pairs': val_warm_pairs,
        'val_cold_pairs': val_cold_pairs,
        'test_warm_pairs': test_warm_pairs,
        'test_cold_pairs': test_cold_pairs,
        'user_to_id': user_to_id,
        'item_to_id': item_to_id,
        'cold_items': cold_items,
        'warm_items': warm_items,
        'num_users': len(users),
        'num_items': len(items)
    }

def build_item_texts(texts_by_item, item_to_meta, item_to_id,
                    max_reviews=MAX_REVIEWS_PER_ITEM, max_chars=MAX_TEXT_CHARS):
    """
    각 아이템의 텍스트 표현 구축

    Returns:
        item_texts: 아이템 인덱스별 텍스트 리스트 매핑
    """
    id_to_item = {v: k for k, v in item_to_id.items()}
    item_texts = {}

    for item_idx in range(len(item_to_id)):
        item_id = id_to_item[item_idx]
        texts = []

        # 메타데이터 텍스트 추가
        meta_text = item_to_meta.get(item_id, "")
        if meta_text:
            texts.append(meta_text[:max_chars])

        # 리뷰 텍스트 추가
        review_texts = texts_by_item.get(item_id, [])
        if review_texts:
            review_texts = review_texts[:max_reviews]
            texts.extend([text[:max_chars] for text in review_texts])

        item_texts[item_idx] = texts if texts else []

    return item_texts

# =============================================================================
# 텍스트 임베딩
# =============================================================================

def create_item_embeddings(item_texts, model_name=SBERT_MODEL, batch_size=TEXT_BATCH):
    """
    SBERT를 사용하여 아이템의 텍스트 임베딩 생성

    Returns:
        v_feat: (아이템 수, 임베딩 차원) 형태의 텐서
    """
    print(f"SBERT 모델 로딩: {model_name}")
    sbert = SentenceTransformer(model_name)

    num_items = len(item_texts)
    embedding_dim = sbert.get_sentence_embedding_dimension()
    embeddings = np.zeros((num_items, embedding_dim), dtype=np.float32)

    def encode_batch(text_list):
        if not text_list:
            return None
        embeddings_batch = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            emb = sbert.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            embeddings_batch.append(emb)
        return np.vstack(embeddings_batch) if embeddings_batch else None

    print(f"{num_items}개 아이템의 임베딩 생성 중...")
    for item_idx in tqdm(range(num_items)):
        texts = item_texts.get(item_idx, [])
        if not texts:
            continue  # 텍스트가 없는 아이템은 0 임베딩 유지

        emb = encode_batch(texts)
        if emb is not None:
            embeddings[item_idx] = emb.mean(axis=0)

    return torch.from_numpy(embeddings)

# =============================================================================
# 데이터셋 및 데이터로더
# =============================================================================

class UIPairDataset(Dataset):
    """사용자-아이템 대조학습용 데이터셋"""

    def __init__(self, ui_pairs, user_pos_items, num_items, neg_per_pos=NEG_PER_POS):
        self.ui_pairs = [(int(u), int(i)) for u, i in ui_pairs]
        self.user_pos_items = user_pos_items
        self.num_items = num_items
        self.neg_per_pos = neg_per_pos
        self.all_items = set(range(num_items))

    def __len__(self):
        return len(self.ui_pairs)

    def __getitem__(self, idx):
        user, pos_item = self.ui_pairs[idx]

        # 음성 샘플 선택
        forbidden = self.user_pos_items[user]
        neg_pool = self.all_items - forbidden

        if len(neg_pool) >= self.neg_per_pos:
            neg_items = np.random.choice(list(neg_pool), size=self.neg_per_pos, replace=False)
        else:
            neg_items = np.random.choice(self.num_items, size=self.neg_per_pos, replace=True)

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long)
        )

class REItemDataset(Dataset):
    """표현 향상 대조학습용 데이터셋"""

    def __init__(self, num_items):
        self.num_items = num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, idx):
        return torch.tensor(idx, dtype=torch.long)

def create_dataloaders(split_data, batch_ui=1024, batch_re=512, num_workers=0):
    """학습용 데이터로더 생성"""
    train_pairs = split_data['train_pairs']
    num_items = split_data['num_items']
    num_users = split_data['num_users']

    # 음성 샘플링을 위한 사용자 양성 아이템 구축
    user_pos_items = [set() for _ in range(num_users)]
    for user, item in train_pairs:
        user_pos_items[int(user)].add(int(item))

    # 데이터셋 생성
    ui_dataset = UIPairDataset(train_pairs, user_pos_items, num_items, NEG_PER_POS)
    re_dataset = REItemDataset(num_items)

    # 데이터로더 생성
    ui_loader = DataLoader(
        ui_dataset, batch_size=batch_ui, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True
    )
    re_loader = DataLoader(
        re_dataset, batch_size=batch_re, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=True
    )

    return ui_loader, re_loader

# =============================================================================
# LightGCN 유틸리티
# =============================================================================

def create_adjacency_matrix(ui_pairs, num_users, num_items):
    """LightGCN용 사용자-아이템 인접 행렬 생성"""
    # 사용자-아이템 상호작용
    rows = [u for u, i in ui_pairs] + [num_users + i for u, i in ui_pairs]
    cols = [num_users + i for u, i in ui_pairs] + [u for u, i in ui_pairs]
    data = [1.0] * (2 * len(ui_pairs))

    # 인접 행렬 생성 [users + items, users + items]
    adj_matrix = sp.coo_matrix((data, (rows, cols)),
                               shape=(num_users + num_items, num_users + num_items))

    # 정규화: D^(-1/2) * A * D^(-1/2)
    adj_matrix = adj_matrix.tocsr()
    rowsum = np.array(adj_matrix.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    norm_adj = d_mat_inv_sqrt @ adj_matrix @ d_mat_inv_sqrt
    return norm_adj.tocoo()

class LightGCN(nn.Module):
    """협업 필터링용 LightGCN 레이어"""

    def __init__(self, num_users, num_items, embedding_dim, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

    def forward(self, user_emb, item_emb, norm_adj_matrix):
        """
        Args:
            user_emb: [사용자 수, 임베딩 차원]
            item_emb: [아이템 수, 임베딩 차원]
            norm_adj_matrix: 정규화된 인접 행렬
        """
        # 사용자와 아이템 임베딩 연결
        all_emb = torch.cat([user_emb, item_emb], dim=0)  # [사용자+아이템 수, 임베딩 차원]
        embs = [all_emb]

        # 그래프 컨볼루션 레이어
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj_matrix, all_emb)
            embs.append(all_emb)

        # 모든 레이어 평균
        final_emb = torch.stack(embs, dim=1).mean(dim=1)

        # 사용자와 아이템 임베딩으로 분할
        user_final = final_emb[:self.num_users]
        item_final = final_emb[self.num_users:]

        return user_final, item_final

# =============================================================================
# 모델 및 손실 함수
# =============================================================================

def info_nce_loss(pos_logits, neg_logits, sample_weights=None):
    """가중 InfoNCE 손실 계산 (선택적 샘플 가중치 지원)"""
    # log(exp(pos) / (exp(pos) + sum(exp(neg)))) 계산
    all_logits = torch.cat([pos_logits[:, None], neg_logits], dim=1)
    log_sum_exp = torch.logsumexp(all_logits, dim=1)
    losses = -(pos_logits - log_sum_exp)  # [배치 크기]

    if sample_weights is not None:
        # 샘플별 가중치 적용 (정규화 포함)
        normalized_weights = sample_weights / (sample_weights.mean() + 1e-8)
        losses = losses * normalized_weights

    return losses.mean()


def score_dot(u_emb, i_emb):
    """내적 점수 함수"""
    return (u_emb * i_emb).sum(dim=-1)

def score_cosine(u_emb, i_emb, eps=1e-8):
    """코사인 유사도 점수 함수"""
    u_norm = F.normalize(u_emb, dim=-1, eps=eps)
    i_norm = F.normalize(i_emb, dim=-1, eps=eps)
    return (u_norm * i_norm).sum(dim=-1)

class CLCRecBaseline(nn.Module):
    """
    표준 임베딩을 사용한 CLCRec 베이스라인 모델
    - U-I: 사용자/아이템 ID 임베딩 + 내적 점수 계산
    - R-E: 콘텐츠 MLP vs 협업 임베딩 정렬
    """

    def __init__(self, num_users, num_items, embedding_dim, feat_dim,
                 re_hidden=128, use_cosine=False, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.use_cosine = use_cosine

        # 사용자-아이템 임베딩 (LightGCN으로 향상됨)
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)

        # LightGCN 레이어
        self.lightgcn = LightGCN(num_users, num_items, embedding_dim, n_layers)

        # 콘텐츠 인코더 (MLP)
        self.content_encoder = nn.Sequential(
            nn.Linear(feat_dim, re_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(re_hidden, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for module in self.content_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_gcn(self, norm_adj_matrix):
        """LightGCN 순전파"""
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        return self.lightgcn(user_emb, item_emb, norm_adj_matrix)

    def ui_score(self, user_idx, item_idx, user_emb_gcn=None, item_emb_gcn=None):
        """LightGCN 향상된 임베딩으로 사용자-아이템 상호작용 점수 계산"""
        if user_emb_gcn is not None and item_emb_gcn is not None:
            # GCN 향상된 임베딩 사용
            user_emb = user_emb_gcn[user_idx]
            item_emb = item_emb_gcn[item_idx]
        else:
            # 원본 임베딩으로 대체
            user_emb = self.user_emb(user_idx)
            item_emb = self.item_emb(item_idx)

        if self.use_cosine:
            return score_cosine(user_emb, item_emb)
        else:
            return score_dot(user_emb, item_emb)

    def get_item_collaborative(self, item_idx, item_emb_gcn=None):
        """아이템의 협업 표현 반환"""
        if item_emb_gcn is not None:
            return item_emb_gcn[item_idx]
        return self.item_emb(item_idx)

    def get_item_content(self, item_idx, item_features):
        """아이템의 콘텐츠 표현 반환"""
        features = item_features[item_idx]
        return self.content_encoder(features)

class CLCRecUncertainty(nn.Module):
    """
    가우시안 임베딩을 사용한 제안 모델
    - U-I: 불확실성 추정이 포함된 가우시안 임베딩
    - R-E: 콘텐츠 MLP vs 협업 임베딩 정렬
    """

    def __init__(self, num_users, num_items, embedding_dim, feat_dim,
                 re_hidden=128, alpha=1e-3, use_cosine=False, n_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.alpha = alpha  # 불확실성 페널티 가중치
        self.use_cosine = use_cosine

        # 가우시안 임베딩 (평균과 로그 분산) - LightGCN으로 향상됨
        self.user_mu = nn.Embedding(num_users, embedding_dim)
        self.user_log_sigma = nn.Embedding(num_users, embedding_dim)  # log(σ)로 양수 보장
        self.item_mu = nn.Embedding(num_items, embedding_dim)
        self.item_log_sigma = nn.Embedding(num_items, embedding_dim)  # log(σ)로 양수 보장

        # LightGCN 레이어
        self.lightgcn = LightGCN(num_users, num_items, embedding_dim, n_layers)

        # 콘텐츠 인코더
        self.content_encoder = nn.Sequential(
            nn.Linear(feat_dim, re_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(re_hidden, embedding_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """가우시안 파라미터 초기화"""
        # 평균 파라미터 초기화
        for emb in [self.user_mu, self.item_mu]:
            nn.init.xavier_uniform_(emb.weight)  # μ: GCN과 동일한 초기화
        # 분산 파라미터 초기화
        for emb in [self.user_log_sigma, self.item_log_sigma]:
            nn.init.constant_(emb.weight, -2.0)  # log(σ) ≈ -2.0, σ ≈ 0.14 (적당한 불확실성)

        # 콘텐츠 인코더 초기화
        for module in self.content_encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_gcn(self, norm_adj_matrix):
        """가우시안 임베딩을 위한 LightGCN 순전파"""
        # GCN 전파를 위해 평균 임베딩 사용
        user_emb = self.user_mu.weight
        item_emb = self.item_mu.weight
        user_gcn, item_gcn = self.lightgcn(user_emb, item_emb, norm_adj_matrix)

        return user_gcn, item_gcn

    def ui_score(self, user_idx, item_idx, user_emb_gcn=None, item_emb_gcn=None):
        """대조학습을 위한 가우시안 사용자-아이템 점수 계산"""
        if user_emb_gcn is not None and item_emb_gcn is not None:
            # GCN 향상된 평균 임베딩 + 원본 분산 사용
            user_mu = user_emb_gcn[user_idx]
            item_mu = item_emb_gcn[item_idx]
            user_log_sigma = self.user_log_sigma(user_idx)
            item_log_sigma = self.item_log_sigma(item_idx)
        else:
            # 원본 임베딩으로 대체
            user_mu = self.user_mu(user_idx)
            item_mu = self.item_mu(item_idx)
            user_log_sigma = self.user_log_sigma(user_idx)
            item_log_sigma = self.item_log_sigma(item_idx)

        # 가우시안 분포에서 임베딩 샘플링
        user_emb = self._sample_gaussian_embedding(user_mu, user_log_sigma, training=self.training)
        item_emb = self._sample_gaussian_embedding(item_mu, item_log_sigma, training=self.training)

        # 기본 상호작용 점수
        base_score = (user_emb * item_emb).sum(dim=-1)
        return base_score

    def _sample_gaussian_embedding(self, mu, log_sigma, training=True):
        """안정적인 가우시안 샘플링"""
        if training:
            # 학습 시: 약간의 노이즈만 추가 (안정성 확보)
            sigma = torch.exp(log_sigma)
            eps = torch.randn_like(mu) * 0.1  # 노이즈 크기 제한
            return mu + sigma * eps
        else:
            # 추론 시: 평균 사용 (일관성 확보)
            return mu

    def compute_uncertainty(self, user_sigma, item_sigma):
        """불확실성 계산 (평균 분산 사용)"""
        # 평균 분산으로 스케일 문제 해결
        user_uncertainty = torch.mean(user_sigma ** 2, dim=-1)  # [배치 크기]
        item_uncertainty = torch.mean(item_sigma ** 2, dim=-1)  # [배치 크기]

        # 상호작용 불확실성 = 사용자 불확실성 + 아이템 불확실성
        interaction_uncertainty = user_uncertainty + item_uncertainty
        return interaction_uncertainty

    def get_item_collaborative(self, item_idx, item_emb_gcn=None):
        """협업 표현 반환 (가우시안 평균 사용)"""
        if item_emb_gcn is not None:
            # GCN 향상된 평균 + 원본 분산 사용
            item_mu = item_emb_gcn[item_idx]
            item_log_sigma = self.item_log_sigma(item_idx)
        else:
            item_mu = self.item_mu(item_idx)
            item_log_sigma = self.item_log_sigma(item_idx)

        # 단순히 평균만 사용 (분산 정보 무시)
        return item_mu

    def get_item_content(self, item_idx, item_features):
        """콘텐츠 표현 반환"""
        features = item_features[item_idx]
        return self.content_encoder(features)

# =============================================================================
# 학습 및 평가
# =============================================================================

def re_inbatch_contrastive_loss(collab_repr, content_repr, temperature=TAU_RE, use_cosine=False, sample_weights=None):
    """표현 향상을 위한 배치 내 대조학습 손실 (선택적 샘플 가중치 지원)"""
    if use_cosine:
        collab_norm = F.normalize(collab_repr, dim=-1)
        content_norm = F.normalize(content_repr, dim=-1)
        logits = collab_norm @ content_norm.t() / temperature
    else:
        logits = (collab_repr @ content_repr.t()) / temperature

    # 대각선 원소는 양성 쌍, 비대각선은 음성 쌍
    batch_size = logits.size(0)
    pos_logits = logits.diag()

    # 음성에서 대각선 원소 제외하는 마스크 생성
    mask = torch.eye(batch_size, device=logits.device).bool()
    neg_logits = logits.masked_fill(mask, float('-inf'))

    # InfoNCE 손실 계산
    log_sum_neg = torch.logsumexp(neg_logits, dim=1)
    losses = -(pos_logits - torch.logsumexp(torch.stack([pos_logits, log_sum_neg], dim=1), dim=1))  # [배치 크기]

    if sample_weights is not None:
        # 샘플별 가중치 적용
        losses = losses * sample_weights

    return losses.mean()

def train_ui_step(model, batch, item_features, norm_adj_matrix, epoch, total_epochs, temperature=TAU_UI):
    """가중 InfoNCE를 사용한 사용자-아이템 대조학습 훈련 단계"""
    user_idx, pos_item, neg_items = batch
    user_idx = user_idx.to(device)
    pos_item = pos_item.to(device)
    neg_items = neg_items.to(device)

    # LightGCN을 통한 순전파
    user_emb_gcn, item_emb_gcn = model.forward_gcn(norm_adj_matrix)

    # 양성 점수
    pos_logits = model.ui_score(user_idx, pos_item, user_emb_gcn, item_emb_gcn) / temperature

    # 음성 점수
    batch_size, num_negs = neg_items.shape
    user_expanded = user_idx.unsqueeze(1).expand(-1, num_negs).reshape(-1)
    neg_expanded = neg_items.reshape(-1)
    neg_logits = model.ui_score(user_expanded, neg_expanded, user_emb_gcn, item_emb_gcn).view(batch_size, -1) / temperature

    # U-CLCRec를 위한 가중 InfoNCE
    if isinstance(model, CLCRecUncertainty):
        # GCN 향상된 임베딩을 사용하여 양성 쌍의 불확실성 계산
        user_mu_gcn = user_emb_gcn[user_idx]  # GCN 향상
        item_mu_gcn = item_emb_gcn[pos_item]  # GCN 향상
        user_sigma = torch.exp(model.user_log_sigma(user_idx))  # 원본 분산
        item_sigma = torch.exp(model.item_log_sigma(pos_item))  # 원본 분산

        # 불확실성 계산
        pos_uncertainty = model.compute_uncertainty(user_sigma, item_sigma)

        # 불확실성 기반 가중치 계산 (불확실성 높을수록 가중치 낮음)
        uncertainty_weight = torch.exp(-model.alpha * pos_uncertainty)

        # 가중 InfoNCE 적용
        ui_loss = info_nce_loss(pos_logits, neg_logits, uncertainty_weight)
    else:
        # 베이스라인 모델: 순수한 InfoNCE
        ui_loss = info_nce_loss(pos_logits, neg_logits)

    return ui_loss

def train_re_step(model, batch, item_features, norm_adj_matrix, epoch, total_epochs, temperature=TAU_RE, use_cosine=False):
    """표현 향상 대조학습 훈련 단계"""
    item_idx = batch.to(device)

    # 협업 표현을 위한 LightGCN 순전파
    user_emb_gcn, item_emb_gcn = model.forward_gcn(norm_adj_matrix)

    # 협업 및 콘텐츠 표현 획득
    collab_repr = model.get_item_collaborative(item_idx, item_emb_gcn)
    content_repr = model.get_item_content(item_idx, item_features)

    # 기본 대조학습 손실
    re_loss = re_inbatch_contrastive_loss(collab_repr, content_repr, temperature, use_cosine)

    return re_loss

def train_epoch(model, ui_loader, re_loader, optimizer, item_features, norm_adj_matrix, epoch, total_epochs):
    """한 에포크 학습"""
    model.train()
    ui_iter = iter(ui_loader)
    re_iter = iter(re_loader)
    num_steps = max(len(ui_loader), len(re_loader))

    total_ui_loss = 0.0
    total_re_loss = 0.0

    pbar = tqdm(range(num_steps), desc=f"Epoch {epoch}/{total_epochs}")

    for step in pbar:
        optimizer.zero_grad()

        # 사용자-아이템 단계
        try:
            ui_batch = next(ui_iter)
        except StopIteration:
            ui_iter = iter(ui_loader)
            ui_batch = next(ui_iter)

        ui_loss = train_ui_step(model, ui_batch, item_features, norm_adj_matrix, epoch, total_epochs)

        # 표현 향상 단계
        try:
            re_batch = next(re_iter)
        except StopIteration:
            re_iter = iter(re_loader)
            re_batch = next(re_iter)

        re_loss = train_re_step(model, re_batch, item_features, norm_adj_matrix, epoch, total_epochs)

        # 결합된 손실
        total_loss = (1 - LAMBDA_RE) * ui_loss + LAMBDA_RE * re_loss
        total_loss.backward()
        optimizer.step()

        total_ui_loss += ui_loss.item()
        total_re_loss += re_loss.item()

        if step % 100 == 0:
            pbar.set_postfix({
                'UI': f'{ui_loss.item():.4f}',
                'RE': f'{re_loss.item():.4f}'
            })

    return total_ui_loss / num_steps, total_re_loss / num_steps

@torch.no_grad()
def compute_all_scores(model, user_batch, item_features, norm_adj_matrix=None):
    """LightGCN과 함께 주어진 사용자들의 모든 아이템 점수 계산"""
    model.eval()
    batch_size = user_batch.size(0)
    num_items = model.num_items

    if isinstance(model, CLCRecUncertainty):
        # 가우시안 모델 - LightGCN과 함께 벡터화된 계산
        user_batch = user_batch.to(device)

        # LightGCN 향상된 임베딩 획득 (가능한 경우)
        if norm_adj_matrix is not None:
            user_emb_gcn, item_emb_gcn = model.forward_gcn(norm_adj_matrix)
            user_mu = user_emb_gcn[user_batch]  # [배치 크기, 임베딩 차원] - GCN 향상
            item_mu = item_emb_gcn  # [아이템 수, 임베딩 차원] - GCN 향상
        else:
            user_mu = model.user_mu(user_batch)  # [배치 크기, 임베딩 차원]
            item_mu = model.item_mu.weight  # [아이템 수, 임베딩 차원]

        # 추론 시에는 평균만 사용 (일관성 확보)
        user_emb = user_mu
        item_emb = item_mu

        # 기본 상호작용 점수 (벡터화)
        base_scores = user_emb @ item_emb.t()  # [배치 크기, 아이템 수]

        # 추론에서는 불확실성 페널티 사용하지 않음 (학습에서만 활용)
        scores = base_scores

        return scores.cpu().numpy()
    else:
        # LightGCN이 포함된 베이스라인 모델
        user_batch = user_batch.to(device)

        if norm_adj_matrix is not None:
            user_emb_gcn, item_emb_gcn = model.forward_gcn(norm_adj_matrix)
            user_emb = user_emb_gcn[user_batch]
            item_emb = item_emb_gcn
        else:
            user_emb = model.user_emb(user_batch)
            item_emb = model.item_emb.weight

        scores = user_emb @ item_emb.t()
        return scores.cpu().numpy()

@torch.no_grad()
def evaluate_model(model, user_pos_dict, item_features, norm_adj_matrix=None, sample_negatives=1000, k=10, measure_time=False):
    """선택적 시간 측정과 함께 모델 성능 평가"""
    if len(user_pos_dict) == 0:
        if measure_time:
            return {"recall@10": 0.0, "ndcg@10": 0.0}, 0.0, 0.0
        else:
            return {"recall@10": 0.0, "ndcg@10": 0.0}

    model.eval()
    users_list = sorted(user_pos_dict.keys())
    batch_size = 512
    all_recalls = []
    all_ndcgs = []
    all_items_set = set(range(model.num_items))
    inference_times = []

    for batch_start in range(0, len(users_list), batch_size):
        batch_users = users_list[batch_start:batch_start + batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long)

        # 요청된 경우 추론 시간 측정
        if measure_time:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

        # 모든 아이템에 대한 점수 획득
        scores = compute_all_scores(model, user_tensor, item_features, norm_adj_matrix)

        if measure_time:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            inference_times.append(end_time - start_time)

        for i, user in enumerate(batch_users):
            pos_items = user_pos_dict[user]
            if len(pos_items) == 0:
                continue

            if sample_negatives is None:
                # 전체 순위
                ranked_items = np.argsort(-scores[i]).tolist()
            else:
                # 음성 샘플링
                neg_pool = list(all_items_set - pos_items)
                if len(neg_pool) >= sample_negatives:
                    neg_items = np.random.choice(neg_pool, size=sample_negatives, replace=False)
                else:
                    neg_items = np.array(neg_pool, dtype=np.int64)

                # 후보 아이템 = 양성 아이템 + 샘플링된 음성 아이템
                candidates = np.concatenate([np.fromiter(pos_items, dtype=np.int64), neg_items])
                candidate_scores = scores[i, candidates]

                # 후보 순위
                top_indices = np.argsort(-candidate_scores)[:k*5]  # 안전을 위해 상위 k*5개
                ranked_items = candidates[top_indices].tolist()

            # 메트릭 계산
            recall = recall_at_k(ranked_items, pos_items, k=k)
            ndcg = ndcg_at_k(ranked_items, pos_items, k=k)

            all_recalls.append(recall)
            all_ndcgs.append(ndcg)

    results = {
        "recall@10": float(np.mean(all_recalls)) if all_recalls else 0.0,
        "ndcg@10": float(np.mean(all_ndcgs)) if all_ndcgs else 0.0
    }

    if measure_time:
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        total_samples = len(users_list)
        avg_batch_size = total_samples / len(inference_times) if len(inference_times) > 0 else 1
        time_per_sample = avg_inference_time / avg_batch_size
        return results, avg_inference_time, time_per_sample
    else:
        return results

def build_user_pos_dict(pairs):
    """쌍에서 사용자 양성 아이템 딕셔너리 구축"""
    user_pos = defaultdict(set)
    for user, item in pairs:
        user_pos[int(user)].add(int(item))
    return dict(user_pos)

def train_and_evaluate_model(model, model_name, split_data, ui_loader, re_loader, item_features, norm_adj_matrix):
    """LightGCN과 함께 완전한 학습 및 평가 파이프라인"""
    print(f"\n{'='*60}")
    print(f"{model_name} 학습 중")
    print(f"{'='*60}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 평가 딕셔너리 구축
    val_warm_dict = build_user_pos_dict(split_data['val_warm_pairs'])
    val_cold_dict = build_user_pos_dict(split_data['val_cold_pairs'])
    test_warm_dict = build_user_pos_dict(split_data['test_warm_pairs'])
    test_cold_dict = build_user_pos_dict(split_data['test_cold_pairs'])

    # 모델 선택을 위한 검증 세트 결합
    val_all_pairs = split_data['val_warm_pairs'] + split_data['val_cold_pairs']
    val_all_dict = build_user_pos_dict(val_all_pairs)

    best_val_score = -1
    best_state_dict = None

    # 학습 루프
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        ui_loss, re_loss = train_epoch(model, ui_loader, re_loader, optimizer,
                                     item_features, norm_adj_matrix, epoch, EPOCHS)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch:2d}/{EPOCHS}: UI Loss={ui_loss:.4f}, RE Loss={re_loss:.4f} ({epoch_time:.1f}s)")

        # 평가
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            val_results = evaluate_model(model, val_all_dict, item_features, norm_adj_matrix,
                                       sample_negatives=1000, k=10)
            val_score = val_results["ndcg@10"]
            print(f"  검증: Recall@10={val_results['recall@10']:.4f}, NDCG@10={val_results['ndcg@10']:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 최고 모델 로드
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 최종 평가 (시간 측정 포함)
    print(f"\n{model_name} 최종 평가:")
    results = {}

    # 시간 측정과 함께 다른 분할에서 테스트
    print("  테스트 세트에서 추론 시간 측정 중...")
    test_all_pairs = split_data['test_warm_pairs'] + split_data['test_cold_pairs']
    test_all_dict = build_user_pos_dict(test_all_pairs)

    # 전체 테스트 세트에서 시간 측정
    test_all_results, avg_time, time_per_sample = evaluate_model(
        model, test_all_dict, item_features, sample_negatives=1000, k=10, measure_time=True
    )
    results['test_all'] = test_all_results
    results['timing'] = {
        'avg_batch_time': avg_time,
        'time_per_sample': time_per_sample
    }

    # 시간 측정 없이 분할에서 테스트 (성능만)
    results['test_warm'] = evaluate_model(model, test_warm_dict, item_features, sample_negatives=1000, k=10)
    results['test_cold'] = evaluate_model(model, test_cold_dict, item_features, sample_negatives=1000, k=10)

    for split_name, metrics in results.items():
        if split_name != 'timing':
            print(f"  {split_name:12s}: Recall@10={metrics['recall@10']:.4f}, NDCG@10={metrics['ndcg@10']:.4f}")

    print(f"  추론 시간: {avg_time*1000:.2f}ms/배치 ({time_per_sample*1000:.3f}ms/샘플)")

    return model, results

# =============================================================================
# 메인 실행
# =============================================================================

def main():
    """메인 실행 파이프라인"""
    print("\n=== 🤖 콜드 스타트 추천시스템 벤치마크 실험 ===")
    print("📊 Baseline vs U-CLCRec 성능 비교")
    print("🎯 목표: 콜드 스타트 아이템에 대한 추천 성능 향상 검증")
    print("="*80)

    # 데이터 로딩
    print("\n=== 📊 데이터 로딩 및 전처리 ===")
    ui_pairs, texts_by_item = load_reviews_jsonl(REVIEWS_PATH)
    item_to_meta = load_meta_jsonl(META_PATH)
    print(f"📥 데이터 로딩 완료:")
    print(f"   • 전체 상호작용: {len(ui_pairs):,}개")
    print(f"   • 리뷰가 있는 아이템: {len(texts_by_item):,}개")
    print(f"   • 메타데이터가 있는 아이템: {len(item_to_meta):,}개")

    # 데이터 분할
    print("\n🔄 콜드/웜 데이터 분할...")
    split_data = create_data_splits(ui_pairs)
    print(f"✅ 분할 완료:")
    print(f"   • 사용자: {split_data['num_users']:,}명")
    print(f"   • 아이템: {split_data['num_items']:,}개 (Cold: {len(split_data['cold_items']):,}, Warm: {len(split_data['warm_items']):,})")
    print(f"   • 학습 데이터: {len(split_data['train_pairs']):,}개")
    print(f"   • 검증 데이터: {len(split_data['val_warm_pairs']) + len(split_data['val_cold_pairs']):,}개")
    print(f"   • 테스트 데이터: {len(split_data['test_warm_pairs']) + len(split_data['test_cold_pairs']):,}개")

    # 텍스트 임베딩 생성
    print("\n🔧 텍스트 임베딩 생성...")
    item_texts = build_item_texts(texts_by_item, item_to_meta, split_data['item_to_id'])
    item_features = create_item_embeddings(item_texts)
    item_features = item_features.to(device)
    feat_dim = item_features.shape[1]

    # 데이터 특성 정보 출력
    print(f"✅ 임베딩 완료: {item_features.shape} (아이템 수 × 임베딩 차원)")
    print(f"📊 원본 특성 정보:")
    print(f"   • 리뷰 특성: summary (요약), reviewText (본문)")
    print(f"   • 메타데이터 특성: title (제목), categories (카테고리)")
    print(f"   • 사용자-아이템 상호작용: reviewerID ↔ asin")
    print(f"   • 텍스트 전처리: summary + reviewText 결합")

    print(f"\n📊 처리된 특성 정보:")
    print(f"   • 최종 임베딩 차원: {feat_dim}개 (SBERT {SBERT_MODEL})")
    print(f"   • 임베딩 방식: 텍스트 → SBERT → 평균 풀링")
    print(f"   • 최대 리뷰 수/아이템: {MAX_REVIEWS_PER_ITEM}개")
    print(f"   • 최대 텍스트 길이: {MAX_TEXT_CHARS}자")

    # 텍스트 데이터 통계
    total_reviews = sum(len(reviews) for reviews in texts_by_item.values())
    items_with_text = len([texts for texts in item_texts.values() if texts])
    items_with_meta = len(item_to_meta)
    print(f"\n📊 데이터 커버리지:")
    print(f"   • 전체 리뷰 개수: {total_reviews:,}개")
    print(f"   • 텍스트가 있는 아이템: {items_with_text:,}개 ({items_with_text/split_data['num_items']*100:.1f}%)")
    print(f"   • 메타데이터가 있는 아이템: {items_with_meta:,}개 ({items_with_meta/split_data['num_items']*100:.1f}%)")

    # 데이터로더 생성
    print("\n📦 데이터로더 생성...")
    ui_loader, re_loader = create_dataloaders(split_data)
    print(f"✅ 데이터로더 준비: UI 배치 {len(ui_loader)}개, RE 배치 {len(re_loader)}개")

    # LightGCN용 인접 행렬 생성
    print("\n📈 LightGCN용 인접 행렬 생성...")
    train_pairs = split_data['train_pairs']
    num_users = split_data['num_users']
    num_items = split_data['num_items']
    norm_adj_matrix = create_adjacency_matrix(train_pairs, num_users, num_items)

    # PyTorch 희소 텐서로 변환
    indices = torch.from_numpy(np.vstack([norm_adj_matrix.row, norm_adj_matrix.col])).long()
    values = torch.from_numpy(norm_adj_matrix.data).float()
    shape = norm_adj_matrix.shape
    norm_adj_matrix = torch.sparse_coo_tensor(indices, values, shape).to(device)
    print(f"✅ 인접 행렬 준비: {shape} 형태, {len(values)} 엣지")

    # 모델 학습 및 평가
    print("\n=== 🚀 모델 학습 및 평가 ===")

    # 베이스라인 모델 (순수한 CLCRec)
    baseline_model = CLCRecBaseline(
        num_users=split_data['num_users'],
        num_items=split_data['num_items'],
        embedding_dim=64,
        feat_dim=feat_dim,
        re_hidden=128,
        use_cosine=False
    )

    baseline_model, baseline_results = train_and_evaluate_model(
        baseline_model, "Baseline", split_data, ui_loader, re_loader, item_features, norm_adj_matrix
    )

    # 제안 모델 (가우시안 임베딩 + 불확실성 학습)
    uncertainty_model = CLCRecUncertainty(
        num_users=split_data['num_users'],
        num_items=split_data['num_items'],
        embedding_dim=64,
        feat_dim=feat_dim,
        re_hidden=128,
        alpha=0.1,
        use_cosine=False
    )

    uncertainty_model, uncertainty_results = train_and_evaluate_model(
        uncertainty_model, "U-CLCRec", split_data, ui_loader, re_loader, item_features, norm_adj_matrix
    )

    # 성능 비교
    print("\n" + "=" * 80)
    print("🎯 콜드 스타트 추천시스템 2개 모델 벤치마크 최종 결과")
    print("🧪 테스트 세트 평가 결과")
    print("=" * 80)

    print(f"\n📊 모델별 성능 (NDCG@10)")
    print(f"   Baseline:      {baseline_results['test_all']['ndcg@10']:.4f}")
    print(f"   U-CLCRec:      {uncertainty_results['test_all']['ndcg@10']:.4f}")

    print(f"\n📊 모델별 성능 (Recall@10)")
    print(f"   Baseline:      {baseline_results['test_all']['recall@10']:.4f}")
    print(f"   U-CLCRec:      {uncertainty_results['test_all']['recall@10']:.4f}")

    # 성능 향상 계산
    def calc_improvement(baseline_val, proposed_val):
        if baseline_val == 0:
            return 0.0
        return ((proposed_val - baseline_val) / baseline_val) * 100

    cold_recall_imp = calc_improvement(
        baseline_results['test_cold']['recall@10'],
        uncertainty_results['test_cold']['recall@10']
    )
    cold_ndcg_imp = calc_improvement(
        baseline_results['test_cold']['ndcg@10'],
        uncertainty_results['test_cold']['ndcg@10']
    )

    warm_recall_imp = calc_improvement(
        baseline_results['test_warm']['recall@10'],
        uncertainty_results['test_warm']['recall@10']
    )
    warm_ndcg_imp = calc_improvement(
        baseline_results['test_warm']['ndcg@10'],
        uncertainty_results['test_warm']['ndcg@10']
    )

    all_recall_imp = calc_improvement(
        baseline_results['test_all']['recall@10'],
        uncertainty_results['test_all']['recall@10']
    )
    all_ndcg_imp = calc_improvement(
        baseline_results['test_all']['ndcg@10'],
        uncertainty_results['test_all']['ndcg@10']
    )

    print(f"\n📈 성능 향상 (U-CLCRec vs Baseline)")
    print(f"   Cold Start - Recall@10:  {cold_recall_imp:+.2f}%")
    print(f"   Cold Start - NDCG@10:    {cold_ndcg_imp:+.2f}%")
    print(f"   Warm Items - Recall@10:  {warm_recall_imp:+.2f}%")
    print(f"   Warm Items - NDCG@10:    {warm_ndcg_imp:+.2f}%")
    print(f"   Overall    - Recall@10:  {all_recall_imp:+.2f}%")
    print(f"   Overall    - NDCG@10:    {all_ndcg_imp:+.2f}%")

    # 추론시간 비교
    baseline_time = baseline_results['timing']['time_per_sample']
    uncertainty_time = uncertainty_results['timing']['time_per_sample']
    baseline_batch_time = baseline_results['timing']['avg_batch_time']
    uncertainty_batch_time = uncertainty_results['timing']['avg_batch_time']

    print(f"\n⏱️ 추론시간 비교")
    print(f"   Baseline:      {baseline_batch_time*1000:.2f}ms/배치 ({baseline_time*1000:.3f}ms/샘플)")
    print(f"   U-CLCRec:      {uncertainty_batch_time*1000:.2f}ms/배치 ({uncertainty_time*1000:.3f}ms/샘플)")

    # 속도 비교
    speed_ratio = uncertainty_time / baseline_time if baseline_time > 0 else 1.0
    print(f"   속도 비교: U-CLCRec는 Baseline의 {speed_ratio:.2f}배 시간 소요")

    # 성능 순위
    ndcg_scores = [
        ("Baseline (Cold)", baseline_results['test_cold']['ndcg@10']),
        ("U-CLCRec (Cold)", uncertainty_results['test_cold']['ndcg@10']),
        ("Baseline (Warm)", baseline_results['test_warm']['ndcg@10']),
        ("U-CLCRec (Warm)", uncertainty_results['test_warm']['ndcg@10']),
    ]

    ndcg_sorted = sorted(ndcg_scores, key=lambda x: x[1], reverse=True)
    print(f"\n🏆 성능 순위 (NDCG@10): {' > '.join([s[0] for s in ndcg_sorted])}")

    # 속도 순위
    speed_scores = [
        ("Baseline", baseline_time),
        ("U-CLCRec", uncertainty_time),
    ]
    speed_sorted = sorted(speed_scores, key=lambda x: x[1])
    print(f"🚀 속도 순위 (빠른 순): {' > '.join([s[0] for s in speed_sorted])}")

    print("\n📊 모델 정보:")
    print(f"   Baseline: {sum(p.numel() for p in baseline_model.parameters()):,} 파라미터")
    print(f"   U-CLCRec: {sum(p.numel() for p in uncertainty_model.parameters()):,} 파라미터")

    print(f"\n📈 학습 결과:")
    print(f"   Baseline: 검증에서 선택된 최적 모델 사용")
    print(f"   U-CLCRec: 검증에서 선택된 최적 모델 사용")

    # 시각화
    print("\n=== 📈 성능 비교 차트 (NDCG@10) ===")
    valid_ndcgs = [
        baseline_results['test_cold']['ndcg@10'],
        uncertainty_results['test_cold']['ndcg@10'],
        baseline_results['test_warm']['ndcg@10'],
        uncertainty_results['test_warm']['ndcg@10'],
        baseline_results['test_all']['ndcg@10'],
        uncertainty_results['test_all']['ndcg@10']
    ]

    if not any(np.isnan(s) for s in valid_ndcgs):
        max_ndcg = max(valid_ndcgs)
        print("NDCG@10 비교:")
        print(f"Cold-Baseline   {'█' * int(baseline_results['test_cold']['ndcg@10']/max_ndcg * 30):<30} {baseline_results['test_cold']['ndcg@10']:.4f}")
        print(f"Cold-U-CLCRec   {'█' * int(uncertainty_results['test_cold']['ndcg@10']/max_ndcg * 30):<30} {uncertainty_results['test_cold']['ndcg@10']:.4f}")
        print(f"Warm-Baseline   {'█' * int(baseline_results['test_warm']['ndcg@10']/max_ndcg * 30):<30} {baseline_results['test_warm']['ndcg@10']:.4f}")
        print(f"Warm-U-CLCRec   {'█' * int(uncertainty_results['test_warm']['ndcg@10']/max_ndcg * 30):<30} {uncertainty_results['test_warm']['ndcg@10']:.4f}")
        print(f"All-Baseline    {'█' * int(baseline_results['test_all']['ndcg@10']/max_ndcg * 30):<30} {baseline_results['test_all']['ndcg@10']:.4f}")
        print(f"All-U-CLCRec    {'█' * int(uncertainty_results['test_all']['ndcg@10']/max_ndcg * 30):<30} {uncertainty_results['test_all']['ndcg@10']:.4f}")

    print("\n=== 📈 성능 비교 차트 (Recall@10) ===")
    valid_recalls = [
        baseline_results['test_cold']['recall@10'],
        uncertainty_results['test_cold']['recall@10'],
        baseline_results['test_warm']['recall@10'],
        uncertainty_results['test_warm']['recall@10'],
        baseline_results['test_all']['recall@10'],
        uncertainty_results['test_all']['recall@10']
    ]

    if not any(np.isnan(s) for s in valid_recalls):
        max_recall = max(valid_recalls)
        print("Recall@10 비교:")
        print(f"Cold-Baseline   {'█' * int(baseline_results['test_cold']['recall@10']/max_recall * 30):<30} {baseline_results['test_cold']['recall@10']:.4f}")
        print(f"Cold-U-CLCRec   {'█' * int(uncertainty_results['test_cold']['recall@10']/max_recall * 30):<30} {uncertainty_results['test_cold']['recall@10']:.4f}")
        print(f"Warm-Baseline   {'█' * int(baseline_results['test_warm']['recall@10']/max_recall * 30):<30} {baseline_results['test_warm']['recall@10']:.4f}")
        print(f"Warm-U-CLCRec   {'█' * int(uncertainty_results['test_warm']['recall@10']/max_recall * 30):<30} {uncertainty_results['test_warm']['recall@10']:.4f}")
        print(f"All-Baseline    {'█' * int(baseline_results['test_all']['recall@10']/max_recall * 30):<30} {baseline_results['test_all']['recall@10']:.4f}")
        print(f"All-U-CLCRec    {'█' * int(uncertainty_results['test_all']['recall@10']/max_recall * 30):<30} {uncertainty_results['test_all']['recall@10']:.4f}")

    print("\n✅ 콜드 스타트 추천시스템 2개 모델 벤치마크 완료!")
    print("📋 비교 모델: Baseline, U-CLCRec")
    print("🎯 Baseline: 표준 임베딩 + U-I/R-E 대조학습")
    print("🎯 U-CLCRec: 가우시안 임베딩 + 불확실성 페널티 + U-I/R-E 대조학습")
    print("🎯 평가 방식: Cold/Warm 아이템 분할 + Recall@10/NDCG@10")
    print("=" * 80)

if __name__ == "__main__":
    main()