from src.rag_benchmark.config import RAGConfig

from src.rag_benchmark.pipeline import EnterpriseRAGPipeline

from datasets import load_dataset

def main():
    config = RAGConfig()


    #Storing the data into vector db

    raw_dataset_document = load_dataset(config.dataset_name,'documents')

    EnPipeLine = EnterpriseRAGPipeline(config)
    cleaned_dataset = EnPipeLine.index_documents(raw_dataset_document['test'])
    print(f"Cleaning and data storing completed")

    # retreiveng the data from vector db
    #test_query = "What is the rollout ramp schedule and time window used for the canary phase when validating adaptive degradation behavior in the Redwood serving runtime?"
    # test_query = "In the enterprise tenant cutover go-live runbook, what latency threshold triggers escalation based on the 95th percentile being too high, and how long does it need to be sustained?"

    # EnPipeLine = EnterpriseRAGPipeline(config)

    # respones = EnPipeLine.query(test_query)
    # return respones
if __name__ == "__main__":
    response = main()
    # print("\n" + "="*50)                                                                                                                                                                  
    # print(f"QUERY: {response['query']}")                                                                                                                                                  
    # print(f"ANSWER:\n{response['answer']}")
    # print(f"CITATIONS: {response['cited_doc_ids']}")
    # print("="*50 + "\n")
  
