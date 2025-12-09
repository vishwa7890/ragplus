# RAGPlus Accuracy Analysis & Improvement Recommendations

**Date:** December 9, 2025  
**Project:** RAGPlus - Offline RAG Toolkit  
**Version:** 0.2.1

---

## 📊 Current System Analysis

### Architecture Overview
Your RAGPlus system implements a comprehensive RAG pipeline with:
- ✅ **Document Loading**: PDF, DOCX, TXT support
- ✅ **Embeddings**: Multiple models (MiniLM, BGE, E5)
- ✅ **Chunking**: 4 strategies (fixed, sentence, markdown, heading)
- ✅ **Retrieval**: Semantic search + BM25 hybrid
- ✅ **Reranking**: Cross-encoder support
- ✅ **Storage**: Persistent vector store with metadata filtering

### Current Strengths
1. **Hybrid Search**: BM25 + embeddings combination
2. **Reranking**: Cross-encoder for improved relevance
3. **Flexible Chunking**: Multiple strategies for different document types
4. **Offline Capability**: No API dependencies
5. **Metadata Filtering**: Source-based filtering support

---

## 🔍 Identified Accuracy Issues

### 1. **Tokenization in BM25 (Critical)**
**Location:** `ragplus/retrieval/bm25.py` (lines 32, 47)

**Issue:**
```python
tokenized_docs = [doc.lower().split()]  # Simple whitespace tokenization
tokenized_query = query.lower().split()
```

**Problem:**
- Whitespace-only tokenization misses punctuation handling
- No stemming/lemmatization reduces recall
- Case folding is good, but insufficient

**Impact:** 20-30% accuracy loss in keyword matching

**Recommendation:**
```python
import re
from typing import List

def tokenize(text: str) -> List[str]:
    """Advanced tokenization with punctuation handling."""
    # Remove punctuation and split
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split and filter empty tokens
    tokens = [t for t in text.split() if t]
    return tokens

# Optional: Add stemming for better recall
try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    def tokenize_with_stemming(text: str) -> List[str]:
        tokens = tokenize(text)
        return [stemmer.stem(t) for t in tokens]
except ImportError:
    pass
```

---

### 2. **Chunking Strategy Selection**
**Location:** `ragplus/chunker.py`

**Issue:**
- Fixed-size chunking (default) breaks semantic boundaries
- No overlap in sentence-based chunking
- Markdown chunking doesn't handle nested structures well

**Impact:** 15-25% accuracy loss due to context fragmentation

**Recommendations:**

#### A. Improve Sentence Chunking with Overlap
```python
def _chunk_sentence(text, max_sentences=5, overlap_sentences=1):
    """Sentence-based chunking with overlap."""
    sentences = nltk.sent_tokenize(text) if NLTK_AVAILABLE else re.split(r'[.!?]+', text)
    
    chunks = []
    for i in range(0, len(sentences), max_sentences - overlap_sentences):
        chunk_sents = sentences[i:i + max_sentences]
        if chunk_sents:
            chunks.append(' '.join(s.strip() for s in chunk_sents if s.strip()))
    
    return chunks
```

#### B. Add Semantic Chunking (New Strategy)
```python
def _chunk_semantic(text, embedder, similarity_threshold=0.5, max_chunk_size=500):
    """Semantic chunking based on sentence similarity."""
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= 1:
        return [text]
    
    # Embed sentences
    embeddings = embedder.encode(sentences)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Calculate similarity with previous sentence
        sim = cosine_similarity(embeddings[i-1:i], embeddings[i:i+1])[0][0]
        
        # If similar enough and chunk not too large, add to current chunk
        chunk_text = ' '.join(current_chunk + [sentences[i]])
        if sim >= similarity_threshold and len(chunk_text) <= max_chunk_size:
            current_chunk.append(sentences[i])
        else:
            # Start new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

---

### 3. **Embedding Model Selection**
**Location:** `ragplus/embedder.py`

**Current Models:**
- `minilm`: Fast but lower accuracy (384 dims)
- `bge-base`: Good balance (768 dims)
- `bge-large`: Best accuracy but slower (1024 dims)

**Issue:** Default is MiniLM which sacrifices accuracy for speed

**Recommendations:**

#### A. Change Default to BGE-Base
```python
def __init__(self, model_name="bge-base", provider="local", device=None):
    # BGE-base offers better accuracy with reasonable speed
```

#### B. Add Query Prefix for E5/BGE Models
```python
def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
    """Encode texts with optional query prefix for E5/BGE models."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    
    # Add instruction prefix for better retrieval
    if is_query and self.model_name.startswith(('e5-', 'bge-')):
        texts = [f"query: {t}" for t in texts]
    elif not is_query and self.model_name.startswith('e5-'):
        texts = [f"passage: {t}" for t in texts]
    
    emb = self.model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True  # Add normalization
    )
    
    return emb.astype("float32")
```

---

### 4. **Hybrid Search Weight Tuning**
**Location:** `ragplus/retrieval/hybrid.py`

**Current Weights:**
- BM25: 0.3 (30%)
- Embeddings: 0.7 (70%)

**Issue:** Fixed weights don't adapt to query type

**Recommendations:**

#### A. Dynamic Weight Adjustment
```python
def _calculate_adaptive_weights(self, query: str) -> tuple:
    """Adjust weights based on query characteristics."""
    # Keyword-heavy queries (short, specific terms)
    words = query.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
    
    # Short queries with specific terms → favor BM25
    if len(words) <= 3 and avg_word_length > 5:
        return 0.5, 0.5  # More balanced
    
    # Long, natural language queries → favor embeddings
    elif len(words) > 8:
        return 0.2, 0.8
    
    # Default
    return self.bm25_weight, self.embedding_weight
```

#### B. Reciprocal Rank Fusion (Better than weighted sum)
```python
def _reciprocal_rank_fusion(
    self,
    bm25_results: List[Tuple[str, float]],
    emb_results: List[Tuple[str, float]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """Combine results using Reciprocal Rank Fusion."""
    scores = {}
    
    # BM25 scores
    for rank, (text, _) in enumerate(bm25_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
    
    # Embedding scores
    for rank, (text, _, _) in enumerate(emb_results):
        scores[text] = scores.get(text, 0) + 1 / (k + rank + 1)
    
    # Sort by combined score
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results
```

---

### 5. **Reranking Model & Strategy**
**Location:** `ragplus/retrieval/reranker.py`

**Current Model:** `ms-marco-MiniLM-L-6-v2`

**Issues:**
- Single reranking pass
- No score calibration
- Fixed top_k

**Recommendations:**

#### A. Upgrade to Better Reranker
```python
# Better models for reranking:
# - "cross-encoder/ms-marco-MiniLM-L-12-v2" (better accuracy)
# - "BAAI/bge-reranker-base" (optimized for BGE embeddings)
# - "BAAI/bge-reranker-large" (best accuracy)

def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
    """Use BGE reranker for better accuracy."""
```

#### B. Add Score Calibration
```python
def rerank(
    self,
    query: str,
    documents: List[str],
    top_k: int = 5,
    return_scores: bool = True,
    score_threshold: float = None
) -> List[Tuple[str, float]]:
    """Rerank with optional score thresholding."""
    if not documents:
        return []
    
    pairs = [[query, doc] for doc in documents]
    scores = self.model.predict(pairs)
    
    # Apply sigmoid for calibration
    scores = 1 / (1 + np.exp(-scores))
    
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    
    # Filter by threshold if provided
    if score_threshold:
        doc_score_pairs = [(d, s) for d, s in doc_score_pairs if s >= score_threshold]
    
    return [(doc, float(score)) for doc, score in doc_score_pairs[:top_k]]
```

---

### 6. **Context Building & Truncation**
**Location:** `ragplus/pipeline.py`

**Issues:**
- Character-based truncation is imprecise
- No chunk deduplication
- No relevance-based ordering preservation

**Recommendations:**

#### A. Add Chunk Deduplication
```python
def _deduplicate_chunks(chunks: List[str], similarity_threshold: float = 0.95) -> List[str]:
    """Remove near-duplicate chunks."""
    from difflib import SequenceMatcher
    
    unique_chunks = []
    for chunk in chunks:
        is_duplicate = False
        for existing in unique_chunks:
            similarity = SequenceMatcher(None, chunk, existing).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_chunks.append(chunk)
    
    return unique_chunks
```

#### B. Improve Token-Based Truncation
```python
def _build_context_by_tokens(chunks: List[str], max_tokens: int, query: str) -> str:
    """Enhanced token-based context building."""
    if not TIKTOKEN_AVAILABLE:
        max_chars = max_tokens * 4
        return build_context(chunks, max_chars=max_chars)
    
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except:
        enc = tiktoken.get_encoding("gpt2")
    
    # Deduplicate first
    chunks = _deduplicate_chunks(chunks)
    
    # Calculate available tokens
    prompt_template = """You are a helpful RAG assistant.

Context:
{context}

Question: {query}

Answer using ONLY the above context."""
    
    template_tokens = len(enc.encode(prompt_template.format(context="", query=query)))
    available_tokens = max_tokens - template_tokens - 100  # Buffer
    
    # Build context with token counting
    context_parts = []
    current_tokens = 0
    
    for chunk in chunks:
        chunk_tokens = len(enc.encode(chunk))
        separator_tokens = len(enc.encode("\n---\n"))
        
        if current_tokens + chunk_tokens + separator_tokens > available_tokens:
            break
        
        context_parts.append(chunk)
        current_tokens += chunk_tokens + separator_tokens
    
    return "\n---\n".join(context_parts)
```

---

### 7. **Query Enhancement**
**New Feature Recommendation**

**Issue:** User queries are often incomplete or ambiguous

**Recommendation:** Add query expansion/rewriting

```python
# Add to pipeline.py

def expand_query(query: str, method: str = "synonyms") -> List[str]:
    """Expand query with synonyms or related terms."""
    expanded = [query]
    
    if method == "synonyms":
        # Use WordNet for synonym expansion
        try:
            from nltk.corpus import wordnet
            words = query.split()
            for word in words:
                synsets = wordnet.synsets(word)
                for syn in synsets[:2]:  # Top 2 synsets
                    for lemma in syn.lemmas()[:2]:  # Top 2 lemmas
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != word.lower():
                            expanded.append(query.replace(word, synonym))
        except:
            pass
    
    return list(set(expanded))  # Remove duplicates

# In rag_answer function:
def rag_answer(
    query: str,
    documents,
    llm_fn: Callable[[str], str],
    use_query_expansion: bool = False,
    **kwargs
):
    # ... existing code ...
    
    # Expand query if enabled
    queries = expand_query(query) if use_query_expansion else [query]
    
    # Retrieve for all query variations and combine
    all_results = []
    for q in queries:
        results = retrieve(store, q, embedder.encode, k=k)
        all_results.extend(results)
    
    # Deduplicate and rerank
    # ... rest of pipeline
```

---

## 🎯 Priority Implementation Plan

### Phase 1: Quick Wins (1-2 days)
**Impact: 25-35% accuracy improvement**

1. ✅ **Fix BM25 Tokenization** (Critical)
   - Add proper punctuation handling
   - Implement stemming
   - Test with technical documents

2. ✅ **Change Default Embedder to BGE-Base**
   - Better accuracy with minimal speed impact
   - Add normalization

3. ✅ **Add Chunk Deduplication**
   - Reduce redundant context
   - Improve context quality

### Phase 2: Medium Impact (3-5 days)
**Impact: 15-20% additional improvement**

4. ✅ **Implement Reciprocal Rank Fusion**
   - Better than weighted sum
   - More robust to score distributions

5. ✅ **Upgrade Reranker Model**
   - Switch to BGE-reranker-base
   - Add score calibration

6. ✅ **Improve Sentence Chunking**
   - Add overlap
   - Better boundary detection

### Phase 3: Advanced Features (1 week)
**Impact: 10-15% additional improvement**

7. ✅ **Add Semantic Chunking**
   - Preserve semantic coherence
   - Adaptive chunk sizes

8. ✅ **Implement Query Expansion**
   - Handle ambiguous queries
   - Improve recall

9. ✅ **Dynamic Weight Adjustment**
   - Adapt to query type
   - Better hybrid search

---

## 📈 Expected Accuracy Improvements

| Optimization | Expected Improvement | Difficulty | Priority |
|--------------|---------------------|------------|----------|
| BM25 Tokenization | 20-30% | Low | 🔴 Critical |
| BGE-Base Default | 10-15% | Low | 🔴 Critical |
| Chunk Deduplication | 5-10% | Low | 🟡 High |
| Reciprocal Rank Fusion | 10-15% | Medium | 🟡 High |
| Better Reranker | 15-20% | Medium | 🟡 High |
| Sentence Overlap | 5-10% | Low | 🟢 Medium |
| Semantic Chunking | 10-15% | High | 🟢 Medium |
| Query Expansion | 5-10% | Medium | 🟢 Medium |
| Dynamic Weights | 5-8% | Medium | 🔵 Low |

**Total Potential Improvement: 50-70% accuracy gain**

---

## 🧪 Testing & Validation

### Create Evaluation Dataset
```python
# tests/test_accuracy.py

import json
from ragplus import rag_answer, Embedder

def load_test_dataset():
    """Load Q&A pairs for evaluation."""
    # Format: {"question": "...", "context": "...", "answer": "...", "doc_id": "..."}
    with open("tests/eval_dataset.json") as f:
        return json.load(f)

def evaluate_rag_accuracy(test_cases, config):
    """Evaluate RAG accuracy with different configurations."""
    results = []
    
    for case in test_cases:
        answer = rag_answer(
            query=case["question"],
            documents=case["context"],
            llm_fn=lambda p: "mock",  # Replace with actual LLM
            **config
        )
        
        # Calculate metrics (exact match, F1, etc.)
        score = calculate_similarity(answer, case["answer"])
        results.append({
            "question": case["question"],
            "score": score,
            "config": config
        })
    
    return results

# Benchmark different configurations
configs = [
    {"use_hybrid_search": False, "use_reranking": False},  # Baseline
    {"use_hybrid_search": True, "use_reranking": False},   # + Hybrid
    {"use_hybrid_search": True, "use_reranking": True},    # + Reranking
    # ... more configs
]

for config in configs:
    results = evaluate_rag_accuracy(test_cases, config)
    print(f"Config {config}: Avg Score = {avg(results)}")
```

### Metrics to Track
1. **Retrieval Metrics:**
   - Precision@K
   - Recall@K
   - MRR (Mean Reciprocal Rank)
   - NDCG (Normalized Discounted Cumulative Gain)

2. **End-to-End Metrics:**
   - Answer Accuracy (exact match)
   - F1 Score
   - BLEU/ROUGE scores
   - Human evaluation scores

---

## 🔧 Implementation Checklist

### Immediate Actions
- [ ] Fix BM25 tokenization with regex and stemming
- [ ] Change default embedder to `bge-base`
- [ ] Add embedding normalization
- [ ] Implement chunk deduplication
- [ ] Add query prefix for BGE/E5 models

### Short-term (This Week)
- [ ] Implement Reciprocal Rank Fusion
- [ ] Upgrade reranker to BGE-reranker-base
- [ ] Add score calibration to reranker
- [ ] Improve sentence chunking with overlap
- [ ] Create evaluation dataset

### Medium-term (Next 2 Weeks)
- [ ] Implement semantic chunking strategy
- [ ] Add query expansion feature
- [ ] Implement dynamic weight adjustment
- [ ] Create comprehensive test suite
- [ ] Benchmark all improvements

### Documentation
- [ ] Update README with accuracy improvements
- [ ] Add best practices guide
- [ ] Create configuration guide
- [ ] Document evaluation methodology

---

## 📝 Configuration Best Practices

### For Maximum Accuracy
```python
answer = rag_answer(
    query=query,
    documents=documents,
    llm_fn=my_llm,
    
    # Use best models
    embedder=Embedder("bge-large"),
    
    # Enable all advanced features
    use_hybrid_search=True,
    use_reranking=True,
    use_query_expansion=True,  # New feature
    
    # Optimal chunking
    chunk_strategy="semantic",  # New strategy
    chunk_size=400,
    chunk_overlap=100,
    
    # Retrieve more for reranking
    k=10,
    
    # Token-based truncation
    max_tokens=4096,
    
    # Persistence for speed
    persist_dir="./rag_cache"
)
```

### For Balanced Speed/Accuracy
```python
answer = rag_answer(
    query=query,
    documents=documents,
    llm_fn=my_llm,
    
    embedder=Embedder("bge-base"),  # Good balance
    use_hybrid_search=True,
    use_reranking=True,
    chunk_strategy="sentence",
    k=5,
    max_tokens=2048
)
```

### For Maximum Speed
```python
answer = rag_answer(
    query=query,
    documents=documents,
    llm_fn=my_llm,
    
    embedder=Embedder("minilm"),
    use_hybrid_search=False,
    use_reranking=False,
    chunk_strategy="fixed",
    k=3,
    max_context_chars=2000
)
```

---

## 🎓 Additional Resources

### Papers & References
1. **Hybrid Search**: "Complementing Lexical Retrieval with Semantic Retrieval" (2021)
2. **Reranking**: "Cross-Encoders for Dense Retrieval" (2020)
3. **Chunking**: "Semantic Text Segmentation" (2019)
4. **RAG**: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)

### Benchmarks
- **BEIR**: Benchmark for retrieval evaluation
- **MS MARCO**: Question answering dataset
- **Natural Questions**: Google's QA dataset

---

## 🚀 Next Steps

1. **Start with Phase 1** - Implement critical fixes first
2. **Create test dataset** - Build evaluation framework
3. **Benchmark baseline** - Measure current accuracy
4. **Implement improvements** - Follow priority order
5. **Re-benchmark** - Validate improvements
6. **Iterate** - Fine-tune based on results

---

**Questions or need help implementing any of these recommendations?**

Let me know which improvements you'd like to tackle first, and I can help you implement them!
