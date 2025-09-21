from ..services.vector_db_agent import VectorDbAgent

def test_retrieve():
    agent = VectorDbAgent()
    results = agent.retrieve_from_qdrant(
        query="What is the salary?",
        collection_name="DOCLING_RAG"
    )
    for hit in results:
        print("Hit : ", hit)

def test_scroll():
    agent = VectorDbAgent()
    results = agent.scroll_from_qdrant(
        collection_name="DOCLING_RAG",
        filters={
            "must": {"source": "Payslip_Apr_2025.pdf"},
            "must_not": {}
        }
    )
    print(results)

if __name__ == "__main__":
    test_retrieve()
    test_scroll()
