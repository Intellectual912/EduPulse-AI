
from src.rag_gemini import load_resources, retrieve_chunks


QUESTIONS = [
    "Explain the OSI reference model and its seven layers.",
    "Describe the five layers of the TCP/IP protocol suite and their functions.",
    "Explain circuit switching and packet switching and their differences.",
    "Explain the three phases of circuit switching: setup, data transfer and teardown.",
    "Explain datagram networks and why they are called connectionless networks.",
    "Explain virtual-circuit networks and their three phases.",
    "Explain the three types of HDLC frames: information, supervisory and unnumbered."
]

TARGET_SUBJECT = "Computer Networks"
TOP_K = 3


def main():
    chunks, embeddings, model, _ = load_resources()

    print("=== HYBRID RETRIEVAL EVALUATION ===")
    print("Total RAG chunks:", len(chunks))
    print("Subject:", TARGET_SUBJECT)
    print("Top results per question:", TOP_K)

    for question in QUESTIONS:
        results = retrieve_chunks(
            question,
            chunks,
            embeddings,
            model,
            subject=TARGET_SUBJECT,
            top_k=TOP_K
        )

        print("\nQUESTION:", question)

        for rank, result in enumerate(results, 1):
            chunk = chunks[result["index"]]

            print(
                f"  Rank {rank} | "
                f"Page {chunk['page']} | "
                f"Semantic {result['semantic_score']:.3f} | "
                f"Keyword {result['keyword_score']:.2f} | "
                f"Fusion {result['fusion_score']:.5f}"
            )

    print("\n=== EVALUATION FINISHED ===")


if __name__ == "__main__":
    main()
