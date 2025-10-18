# 🔮 U-CLCRec: Uncertainty-aware Contrastive Learning for Cold-start Recommendation

### 🧠 A graph-based recommender system that models uncertainty with Gaussian embeddings and dual contrastive learning.

---

## 🚀 Overview
**U-CLCRec** is a recommender system designed to solve the **cold-start problem** — recommending new items with minimal interaction data.  
It combines **LightGCN-based collaborative learning**, **Sentence-BERT text embeddings**, and **uncertainty-aware contrastive learning** to provide robust, reliable recommendations.

---

## 🎯 Key Features
- 🧩 **Dual Contrastive Learning**  
  - *User–Item (U–I)* contrastive learning  
  - *Representation–Enhancement (R–E)* contrastive learning  
- 🌫️ **Gaussian Embeddings** — Model uncertainty via learnable mean (μ) and variance (σ²)  
- 🔗 **LightGCN Integration** — Capture high-order collaborative signals through graph propagation  
- 🧠 **Uncertainty-weighted Loss** — Downweight uncertain or noisy samples during training  
- 📊 **Cold/Warm Evaluation** — Separate benchmarking for unseen vs. known items  

---

## 🏗️ Architecture
```text
User–Item Graph ──► LightGCN ─┐
                               │
                               ├──► User–Item Contrastive Loss (L_UI)
Text Reviews & Metadata ─► SBERT ─┘
           │
           └──► Representation–Enhancement Loss (L_RE)
Final Loss: (1 - λ)L_UI + λL_RE  (weighted by uncertainty)
```

---

## 📁 Project Structure
```text
U-CLCRec/
├── cold_start_recommender.py      # Main training & evaluation pipeline
├── data/                          # Review and metadata files
│   ├── Magazine_Subscriptions.json
│   └── meta_Magazine_Subscriptions.json
├── embeddings/                    # SBERT text embeddings
└── README.md
```

---

## 📈 Experimental Results

📊 **NDCG@10 Comparison**
| Split | Baseline | U-CLCRec | Improvement |
|--------|-----------|-----------|--------------|
| **Cold** | 0.0103 | **0.0247** | +139.8% |
| **Warm** | 0.0979 | **0.1069** | +9.2% |
| **All** | 0.0566 | **0.0669** | +18.2% |

📊 **Recall@10 Comparison**
| Split | Baseline | U-CLCRec | Improvement |
|--------|-----------|-----------|--------------|
| **Cold** | 0.0228 | **0.0497** | +118.0% |
| **Warm** | 0.1344 | **0.1459** | +8.5% |
| **All** | 0.0808 | **0.0983** | +21.7% |

> 🧩 **U-CLCRec** significantly improves performance for cold-start items  
> while maintaining stable accuracy on warm items.

---

## ⚡ Inference Speed

| Model | Time per Batch | Time per Sample | Relative Speed |
|--------|----------------|------------------|----------------|
| **CLCRec (Baseline)** | 2.01 ms | 0.004 ms | 1.00× |
| **U-CLCRec (with Uncertainty)** | **1.75 ms** | **0.003 ms** | **0.87× faster** |

> ⚙️ Despite additional uncertainty modeling, U-CLCRec runs *faster* than the baseline.

---

## 🛠️ Tech Stack
| Category | Library |
|-----------|----------|
| Framework | PyTorch |
| Text Encoder | Sentence-BERT (`all-MiniLM-L6-v2`) |
| Graph Model | LightGCN |
| Optimizer | AdamW |
| Metrics | Recall@10, NDCG@10 |
| Environment | Google Colab (GPU) |

---

## 💡 Research Highlights
- Proposed **uncertainty-weighted InfoNCE loss** for robust recommendation learning  
- Modeled users and items as **probabilistic Gaussian embeddings**  
- Unified **content-based** and **collaborative** signals in a dual contrastive setup  
- Designed **cold/warm evaluation split** reflecting real-world recommendation scenarios  

---

## 🧭 Future Work
- Variational LightGCN for probabilistic message passing  
- Real-time cold-start adaptation with online learning  
- Multimodal recommendation (text + image + audio)  

---

## 📚 References
- He et al., *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*, SIGIR 2020  
- Gao et al., *Contrastive Learning for Cold-start Recommendation*, RecSys 2022  
- Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019  

---

**Author:** [Mincheol Shin]  
**License:** MIT  