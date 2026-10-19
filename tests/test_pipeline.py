"""
Integration test for the EnterpriseRAGPipeline end-to-end.
"""
import shutil
from rag_benchmark import EnterpriseRAGPipeline, RAGConfig


def test_enterprise_rag_pipeline_end_to_end():
    test_db_dir = "data/test_qdrant_db"
    shutil.rmtree(test_db_dir, ignore_errors=True)

    config = RAGConfig(
        storage_path=test_db_dir,
        collection_name="test_enterprise_chunks",
        hybrid_top_k=5,
        rerank_top_n=2,
        generator_model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    pipeline = EnterpriseRAGPipeline(config=config)

    sample_docs = [
        {
            "doc_id": "dsid_runbook_canary",
            "source_type": "confluence",
            "title": "Production Canary Deployment SOP",
            "content": (
                "Canary Deployment SOP for Redwood Inference Service:\n"
                "1. Deploy new image to 5% canary traffic slice.\n"
                "2. Monitor Datadog p99 latency and HTTP 5xx error rate.\n"
                "3. If 5xx error rate exceeds 0.5% over a 5-minute rolling window, "
                "trigger an immediate rollback using: kubectl rollout undo deployment/redwood-api.\n"
                "4. Post incident report to #eng-incidents Slack channel."
            ),
        },
        {
            "doc_id": "dsid_slack_lunch",
            "source_type": "slack",
            "title": "Team Lunch Discussion",
            "content": "Hey everyone, catering today is Mexican food at 12:30 PM in the 4th floor lounge.",
        },
        {
            "doc_id": "dsid_jira_bug123",
            "source_type": "jira",
            "title": "JIRA-123: Memory leak in worker pod",
            "content": "Investigating high resident memory in Celery workers. Fixed by setting max-tasks-per-child=100.",
        },
    ]

    # 1. Test Ingestion & Indexing
    stats = pipeline.index_documents(sample_docs)
    print("\nIndexing stats:", stats)
    assert stats["documents_received"] == 3
    assert stats["documents_indexed"] == 3
    assert stats["child_chunks"] >= 3
    assert stats["points_upserted"] >= 3

    # 2. Test Querying & Grounded Generation
    query = "What should I do if the canary 5xx error rate exceeds 0.5%?"
    response = pipeline.query(query)

    print("\nQuery:", response["query"])
    print("\nAnswer:\n", response["answer"])
    print("\nCited IDs:", response["cited_doc_ids"])

    assert len(response["retrieved_chunks"]) > 0
    top_doc_id = response["retrieved_chunks"][0]["doc_id"]
    assert top_doc_id == "dsid_runbook_canary", f"Expected dsid_runbook_canary top doc, got {top_doc_id}"
    assert "dsid_runbook_canary" in response["cited_doc_ids"] or "kubectl rollout undo" in response["answer"]

    # Clean up test database
    pipeline.close()
    shutil.rmtree(test_db_dir, ignore_errors=True)
    print("\nEnd-to-end integration test completed successfully!")


if __name__ == "__main__":
    test_enterprise_rag_pipeline_end_to_end()
