from ragplus import rag_answer, Embedder

def dummy_llm(prompt: str) -> str:
    """Mock LLM that just returns the prompt."""
    return "This is a mock LLM.\n\nPrompt:\n" + prompt

docs = [
    "Vacuum brazing uses a filler metal in a furnace.",
    "The process gives clean joints and low distortion."
]

query = "Explain vacuum brazing"

# Debug: Print document lengths
print(len(docs[0]), len(docs[1]))

embedder = Embedder()
ans = rag_answer(
    query,
    docs,
    llm_fn=dummy_llm,
    embedder=embedder,
    chunk_size=300,
    chunk_overlap=50
)

print(ans)
