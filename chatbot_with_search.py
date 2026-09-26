from __future__ import annotations

from duckduckgo_search import DDGS
from transformers import pipeline, set_seed

MODEL_NAME = "google/flan-t5-base"


def web_search(query: str, max_results: int = 3) -> str:
    """Return the top web search snippets for a query, joined into one string."""
    try:
        results = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        return f"error: web search failed ({exc})"

    snippets = [r.get("body", "") for r in results if r.get("body")]
    return " ".join(snippets) or "No search results found."


def build_prompt(question: str, context: str) -> str:
    """RAG prompt: ask the model to answer using only the retrieved context."""
    return (
        "Answer the question using only the context below.\n"
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def main() -> None:
    set_seed(42)  # reproducible demo output
    generator = pipeline("text2text-generation", model=MODEL_NAME)

    print("Ask a question (type 'exit' or 'quit' to stop).")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        context = web_search(question)
        print(f"[search context]: {context[:200]}{'...' if len(context) > 200 else ''}")

        prompt = build_prompt(question, context)
        answer = generator(prompt, max_new_tokens=60)[0]["generated_text"].strip()
        print(f"LLM answer (grounded in search): {answer}")


if __name__ == "__main__":
    main()