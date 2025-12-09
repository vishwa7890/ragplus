---
description: Workflow to improve RAG accuracy with prioritized fixes
---

# RAGPlus Accuracy Improvement Workflow

## Phase 1: Critical Fixes (Immediate - 25-35% improvement)

### 1. Fix BM25 Tokenization
**File:** `ragplus/retrieval/bm25.py`
**Impact:** 20-30% accuracy improvement

```bash
# Test current BM25 performance
python -c "from ragplus.retrieval import BM25Retriever; print('BM25 loaded')"
```

**Changes needed:**
- Replace simple `split()` with regex-based tokenization
- Add punctuation handling
- Implement optional stemming
- Add stopword removal

### 2. Upgrade Default Embedder
**File:** `ragplus/embedder.py`
**Impact:** 10-15% accuracy improvement

**Changes needed:**
- Change default from `minilm` to `bge-base`
- Add `normalize_embeddings=True`
- Add query/passage prefixes for BGE/E5 models
- Update `encode()` method signature

### 3. Add Chunk Deduplication
**File:** `ragplus/pipeline.py`
**Impact:** 5-10% accuracy improvement

**Changes needed:**
- Create `_deduplicate_chunks()` function
- Integrate into context building
- Use SequenceMatcher for similarity detection

## Phase 2: High-Impact Features (3-5 days - 15-20% improvement)

### 4. Implement Reciprocal Rank Fusion
**File:** `ragplus/retrieval/hybrid.py`
**Impact:** 10-15% accuracy improvement

**Changes needed:**
- Add `_reciprocal_rank_fusion()` method
- Replace weighted sum with RRF
- Tune k parameter (default: 60)

### 5. Upgrade Reranker
**File:** `ragplus/retrieval/reranker.py`
**Impact:** 15-20% accuracy improvement

**Changes needed:**
- Change default model to `BAAI/bge-reranker-base`
- Add score calibration (sigmoid)
- Add score thresholding option
- Improve error handling

### 6. Improve Sentence Chunking
**File:** `ragplus/chunker.py`
**Impact:** 5-10% accuracy improvement

**Changes needed:**
- Add overlap to sentence chunking
- Improve boundary detection
- Better handling of edge cases

## Phase 3: Advanced Features (1 week - 10-15% improvement)

### 7. Add Semantic Chunking
**File:** `ragplus/chunker.py`
**Impact:** 10-15% accuracy improvement

**Changes needed:**
- Create `_chunk_semantic()` function
- Use embeddings to detect topic boundaries
- Adaptive chunk sizing

### 8. Implement Query Expansion
**File:** `ragplus/pipeline.py`
**Impact:** 5-10% accuracy improvement

**Changes needed:**
- Create `expand_query()` function
- Add synonym expansion using WordNet
- Integrate into retrieval pipeline

### 9. Dynamic Weight Adjustment
**File:** `ragplus/retrieval/hybrid.py`
**Impact:** 5-8% accuracy improvement

**Changes needed:**
- Create `_calculate_adaptive_weights()` method
- Analyze query characteristics
- Adjust BM25/embedding weights dynamically

## Testing & Validation

### Create Test Dataset
```bash
# Create evaluation dataset
mkdir -p tests/data
# Add test cases with ground truth
```

### Run Benchmarks
```bash
# Baseline
python tests/test_accuracy.py --config baseline

# After Phase 1
python tests/test_accuracy.py --config phase1

# After Phase 2
python tests/test_accuracy.py --config phase2

# After Phase 3
python tests/test_accuracy.py --config phase3
```

### Metrics to Track
- Precision@5, Recall@5
- MRR (Mean Reciprocal Rank)
- NDCG@10
- End-to-end answer accuracy

## Execution Order

1. **Day 1:** Fix BM25 tokenization + Upgrade embedder
2. **Day 2:** Add deduplication + Test Phase 1
3. **Day 3:** Implement RRF + Upgrade reranker
4. **Day 4:** Improve sentence chunking + Test Phase 2
5. **Day 5:** Add semantic chunking
6. **Day 6:** Implement query expansion
7. **Day 7:** Add dynamic weights + Final testing

## Success Criteria

- [ ] All tests pass
- [ ] Accuracy improved by at least 50%
- [ ] No performance regression (< 20% slower)
- [ ] Documentation updated
- [ ] Examples updated with best practices
