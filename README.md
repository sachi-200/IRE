# NewsRecX — Personalized News Recommendation System

An end-to-end news recommendation system for the **MIND** and **EB-NeRD** datasets. The project combines lexical retrieval, semantic retrieval, behavioural features, learning-to-rank, neural news recommendation, evaluation, and large-scale prediction generation into a single reproducible pipeline.

The system follows a two-stage recommendation architecture:

1. **Candidate Retrieval** — retrieve relevant news articles using BM25 and semantic embeddings.
2. **Candidate Re-ranking** — combine retrieval signals with behavioural, session, article, and positional features using a LightGBM LambdaMART re-ranker.

A neural **NRMS** baseline is also reproduced and extended with a category-aware news representation.


# Overview

News recommendation requires ranking articles according to the probability that a user will interact with them.

This project builds a complete recommendation pipeline using:

- User click history
- Article titles and abstracts
- Semantic article representations
- BM25 lexical relevance
- Session behaviour
- Article popularity
- Article freshness
- Category similarity
- Position bias
- Dwell time where available
- Learning-to-rank
- Neural self-attention based recommendation

The system operates on:

- **MIND** — Microsoft News Dataset
- **EB-NeRD** — Ekstra Bladet News Recommendation Dataset

The retrieval layer provides high-recall candidates, while the ranking layer incorporates behavioural signals to determine the final ordering.


# Key Features

## Retrieval

- BM25 lexical retrieval
- Multilingual semantic embeddings
- Exact cosine similarity search
- FAISS-compatible vector indexing
- User representations constructed from recent click history
- Hybrid retrieval support

## Behavioural Modelling

- Historical click count
- Recency-weighted click count
- Cold-start identification
- Article popularity
- Article freshness
- Category matching
- Session statistics
- Dwell time
- Retrieval position
- Position-based click-through-rate prior

## Learning-to-Rank

- LightGBM LambdaMART
- 14 ranking features
- Positive/negative candidate sampling
- Candidate re-ranking after retrieval

## Neural Recommendation

- NRMS-style news encoder
- Multi-head self-attention
- Additive attention pooling
- User encoder based on recent article representations
- Category-aware extension
- Paired bootstrap significance testing

## Evaluation

- AUC
- MRR
- nDCG@5
- nDCG@10
- Intra-list diversity
- Novelty
- Catalog coverage
- Cold-start vs. warm-user analysis
- Head vs. tail analysis
- Bootstrap 95% confidence intervals

## Production-Oriented Analysis

- Index memory measurement
- Feature-store memory measurement
- p50/p99 latency
- Throughput
- Cost per 1,000 queries
- 10× traffic analysis
- 10× catalog-size analysis


# System Architecture

The complete system follows a retrieve-then-rank architecture:

```text
                         ┌─────────────────────┐
                         │     Raw Datasets    │
                         │   MIND / EB-NeRD    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Data Processing    │
                         │  Cleaning + Parsing  │
                         │  Temporal Splitting  │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
        ┌──────────────────┐                  ┌──────────────────┐
        │  BM25 Retriever  │                  │ Semantic Encoder│
        │                  │                  │                  │
        │ Title + Abstract │                  │ Sentence         │
        │ Sparse Index     │                  │ Embeddings       │
        └────────┬─────────┘                  └────────┬─────────┘
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Candidate Retrieval │
                         │       Top-K         │
                         │      K = 150        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │     Behavioural Feature Store    │
                  │                                  │
                  │ Click history                    │
                  │ Recency                          │
                  │ Popularity                       │
                  │ Freshness                        │
                  │ Category matching                │
                  │ Session statistics               │
                  │ Position features                │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │ LightGBM LambdaMART │
                         │     Re-ranker       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Final Ranked List   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Evaluation /        │
                         │ Prediction Output   │
                         └─────────────────────┘
```


# Datasets

## MIND

The Microsoft News Dataset contains English news articles and user interaction logs.

The project uses:

- Article metadata
- Titles
- Abstracts
- User behaviours
- Click history
- Candidate impressions
- Category/entity information where available

The large MIND test set contains approximately **2.37 million impressions**.

## EB-NeRD

EB-NeRD is a large-scale Danish news recommendation dataset from Ekstra Bladet.

The dataset contains:

- News articles
- User interactions
- Impression logs
- Session information
- Read/dwell-time information
- Pre-computed article embeddings

The large EB-NeRD test set contains approximately **13.5 million impressions**.

For development, smaller/demo datasets can be used before running the large-scale pipeline.


# Pipeline

## 1. Data Processing

Raw data from MIND and EB-NeRD is converted into a unified representation containing:

- Articles
- Impressions
- Candidate articles
- Click history
- User features
- Article features

Interaction data is split chronologically rather than randomly.

This is important because news recommendation is highly temporal: information available in the future must never be used to construct features for an earlier impression.

The resulting feature store contains reusable article and user representations.


# 2. Lexical Retrieval

The lexical retrieval component uses **BM25** over article titles and abstracts.

The BM25 implementation uses:

```text
k1 = 1.5
b  = 0.75
```

For each user, the retrieval query is constructed using the titles and abstracts of the user's **five most recently clicked articles**.

Conceptually:

```text
User click history
        │
        ▼
Last 5 clicked articles
        │
        ▼
Concatenate title + abstract
        │
        ▼
BM25 query
        │
        ▼
Top-K candidate articles
```

The implementation also provides vectorized scoring for efficient batch processing.


# 3. Semantic Retrieval

Semantic retrieval represents articles and users in a dense embedding space.

The system uses:

```text
sentence-transformers/
multilingual-MiniLM-L12-v2
```

The model provides a shared multilingual representation for both:

- English MIND articles
- Danish EB-NeRD articles

The user's representation is computed by averaging the embeddings of the five most recently clicked articles and re-normalizing the resulting vector.

Semantic similarity is computed using cosine similarity.

For the datasets used during development, exact/brute-force search or a FAISS flat index is sufficient.


# 4. Behavioural Feature Engineering

The ranking model uses behavioural information derived from historical click logs.

All features are computed **point-in-time**.

For an impression occurring at time `t`, only information available strictly before `t` can be used.

## Click-history features

| Feature | Description |
|---|---|
| `n_clicks_before` | Number of clicks before the current impression |
| `recency_weighted_click_count` | Click count weighted using exponential time decay |
| `is_cold_start` | Indicates whether the user has limited historical activity |

## Article features

| Feature | Description |
|---|---|
| `popularity_prior_ctr` | Historical article click-through-rate prior |
| `freshness_hours` | Time since article publication |
| `category_match` | Whether article category matches user interests |
| `category_match_frac` | Fraction of historical categories matching the candidate |

## Session features

| Feature | Description |
|---|---|
| `session_click_count_before` | Number of clicks before the impression in the session |
| `session_impressions_before` | Previous impressions in the session |
| `avg_dwell_time_before` | Historical average dwell time |

EB-NeRD provides native session and read-time information.

For MIND, sessions are derived using a **30-minute inactivity threshold**, while dwell time is left unavailable rather than being silently imputed.

## Position features

| Feature | Description |
|---|---|
| `position` | Candidate's retrieval position |
| `position_ctr_prior` | Historical CTR associated with candidate position |


# 5. Learning-to-Rank

The final ranking stage uses **LightGBM LambdaMART**.

The model receives 14 features:

```text
2 retrieval features
+
12 behavioural features
=
14 total features
```

The retrieval features are:

```text
bm25_score
semantic_score
```

The remaining 12 features are behavioural, article, session, and position features described above.

The retrieval stage retains up to:

```text
K = 150
```

candidates.

The LightGBM model then scores these candidates and produces the final recommendation order.

Training uses:

- One positive candidate
- Up to four sampled negatives
- LambdaMART ranking objective

At inference time, the complete retained candidate set is scored.


# 6. NRMS Baseline and Category-Aware Model

The project also reproduces an **NRMS-style neural news recommendation model**.

The architecture contains two main components.

## News Encoder

```text
Title tokens
      │
      ▼
Word Embeddings
      │
      ▼
Multi-Head Self-Attention
      │
      ▼
Additive Attention Pooling
      │
      ▼
News Representation
```

## User Encoder

The user's recent news representations are passed through the same general attention mechanism:

```text
Recent clicked news
        │
        ▼
Self-Attention
        │
        ▼
Additive Attention
        │
        ▼
User Representation
```

The user and candidate representations are compared using a dot product.

The neural model is trained using:

- One positive article
- Four sampled negatives
- Pointwise binary cross-entropy

## Category-Aware Extension

The category-aware model adds a learned category embedding to the news representation.

The category representation is:

```text
Category Embedding
       │
       ├──────────────┐
       │              │
       ▼              ▼
Title Attention   Category Vector
       │              │
       └──────┬───────┘
              ▼
       Concatenation
              │
              ▼
          Projection
              │
              ▼
       News Representation
```

The user encoder, scoring function, training data, hyperparameters, and random seed are kept unchanged so that the category representation can be evaluated as an isolated architectural change.


# 7. Evaluation

The project evaluates recommendation quality using both accuracy and beyond-accuracy metrics.

## Ranking Metrics

### AUC

Measures the ability of the model to distinguish clicked from non-clicked candidates.

### MRR

Measures how highly the first relevant article is ranked.

### nDCG@5

Measures ranking quality in the top five recommendations.

### nDCG@10

Measures ranking quality in the top ten recommendations.

## Beyond-Accuracy Metrics

### Intra-list Diversity

Measures how different the recommended articles are from one another.

### Novelty

Measures how much the system recommends less-popular articles.

### Catalog Coverage

Measures the fraction of the complete article catalog that appears in recommendations.

## Evaluation Slices

### Cold-start vs. Warm

Users are divided according to their available historical click activity.

### Head vs. Tail

Articles are divided using training-set popularity.

The head consists of articles at or above the 80th percentile of training popularity.

## Statistical Significance

Bootstrap confidence intervals are used for reported metrics.

For model comparisons, paired bootstrap analysis is performed over per-impression metric differences.

A claimed improvement is considered statistically supported when its 95% confidence interval excludes zero in the claimed direction.
