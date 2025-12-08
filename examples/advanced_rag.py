"""
Advanced RAG example demonstrating v0.2.1 features:
- Document loading (PDF, DOCX, TXT)
- Multiple embedding models (BGE, E5)
- Hybrid search (BM25 + embeddings)
- Cross-encoder reranking
- Persistent vector store
- Metadata filtering
- Semantic chunking
"""

from ragplus import (
    load_document,
    Embedder,
    VectorStore,
    HybridRetriever,
    Reranker,
    chunk_text,
    rag_answer
)


def demo_document_loading():
    """Demo: Load various document formats."""
    print("=== Document Loading Demo ===\n")
    
    # Load different file types
    # text = load_document("sample.pdf")
    # text = load_document("report.docx")
    text = "Sample document text for demonstration purposes."
    
    print(f"Loaded text (first 200 chars): {text[:200]}...\n")
    return text


def demo_embedding_models():
    """Demo: Use different embedding models."""
    print("=== Embedding Models Demo ===\n")
    
    # Use BGE model (higher accuracy)
    embedder_bge = Embedder(model_name="bge-base")
    print(f"Using model: {embedder_bge.model_name}")
    
    # Use E5 model
    # embedder_e5 = Embedder(model_name="e5-base")
    
    # Use default MiniLM
    # embedder_mini = Embedder(model_name="minilm")
    
    return embedder_bge


def demo_semantic_chunking():
    """Demo: Semantic chunking strategies."""
    print("\n=== Semantic Chunking Demo ===\n")
    
    markdown_text = """
# Introduction
This is the introduction section.

## Background
Some background information here.

## Methods
Description of methods used.
"""
    
    # Markdown-aware chunking
    chunks = chunk_text(markdown_text, strategy="markdown")
    print(f"Markdown chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: {chunk[:50]}...")
    
    # Sentence-based chunking
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    sent_chunks = chunk_text(text, strategy="sentence", size=3)
    print(f"\nSentence chunks: {len(sent_chunks)}")
    
    return chunks


def demo_hybrid_search():
    """Demo: Hybrid search with BM25 + embeddings."""
    print("\n=== Hybrid Search Demo ===\n")
    
    # Sample documents
    docs = [
        "Python is a high-level programming language.",
        "Machine learning uses algorithms to learn from data.",
        "RAG combines retrieval with generation for better answers.",
        "Vector databases store embeddings for similarity search."
    ]
    
    # Create embedder and vectorstore
    embedder = Embedder(model_name="minilm")
    vectors = embedder.encode(docs)
    
    store = VectorStore()
    store.add(docs, vectors)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        vectorstore=store,
        embedder=embedder,
        bm25_weight=0.3,
        embedding_weight=0.7
    )
    
    # Search
    query = "What is RAG?"
    results = hybrid.search(query, k=2)
    
    print(f"Query: {query}")
    print("Results:")
    for text, meta, score in results:
        print(f"  - {text} (score: {score:.3f})")
    
    return hybrid


def demo_reranking():
    """Demo: Cross-encoder reranking."""
    print("\n=== Reranking Demo ===\n")
    
    query = "machine learning algorithms"
    documents = [
        "Python is a programming language",
        "Machine learning uses algorithms to learn patterns",
        "Databases store structured data",
        "Neural networks are a type of ML algorithm"
    ]
    
    # Rerank documents
    reranker = Reranker()
    reranked = reranker.rerank(query, documents, top_k=2)
    
    print(f"Query: {query}")
    print("Top results after reranking:")
    for doc, score in reranked:
        print(f"  - {doc} (score: {score:.3f})")


def demo_persistent_storage():
    """Demo: Persistent vector store."""
    print("\n=== Persistent Storage Demo ===\n")
    
    # Create store with persistence
    store = VectorStore(persist_dir="./ragplus_data")
    
    # Add documents
    docs = ["Document 1", "Document 2"]
    embedder = Embedder()
    vectors = embedder.encode(docs)
    
    store.add_documents(docs, vectors, doc_id="demo.txt")
    
    print("Documents saved to ./ragplus_data")
    print("Store will auto-load on next initialization")


def demo_metadata_filtering():
    """Demo: Metadata filtering."""
    print("\n=== Metadata Filtering Demo ===\n")
    
    # Create store with metadata
    docs = ["Doc from file1", "Doc from file2", "Another from file1"]
    metas = [
        {"source": "file1.pdf", "page": 1},
        {"source": "file2.pdf", "page": 1},
        {"source": "file1.pdf", "page": 2}
    ]
    
    embedder = Embedder()
    vectors = embedder.encode(docs)
    
    store = VectorStore()
    store.add(docs, vectors, metas)
    
    # Search with filter
    query_vec = embedder.encode(["document"])
    results = store.search(query_vec, k=5, filter={"source": "file1.pdf"})
    
    print("Results filtered by source='file1.pdf':")
    for text, meta, score in results:
        print(f"  - {text} (page: {meta['page']})")


def main():
    """Run all demos."""
    print("RAGPlus v0.2.1 - Advanced Features Demo\n")
    print("=" * 50)
    
    try:
        demo_document_loading()
        demo_embedding_models()
        demo_semantic_chunking()
        demo_hybrid_search()
        demo_reranking()
        demo_persistent_storage()
        demo_metadata_filtering()
        
        print("\n" + "=" * 50)
        print("All demos completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Some features may require additional dependencies.")
        print("Install with: pip install ragplus[all]")


if __name__ == "__main__":
    main()
