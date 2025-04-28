# evaluation.py

import time
import pandas as pd
import retrieval_functions as rf

# Load gold test set
questions = pd.read_excel('gold_standard_test_set.xlsx')

# Define retrieval methods
retrieval_methods = {
    "FAISS": lambda q: rf.query_faiss_text(q, top_k=1),
    "pgvector-IVFFLAT": lambda q: rf.query_pgvector(q, method="ivfflat", top_k=1),
    "pgvector-HNSW": lambda q: rf.query_pgvector(q, method="hnsw", top_k=1),
    "TF-IDF": lambda q: rf.query_tfidf(q, top_k=1),
    "BM25": lambda q: rf.query_bm25(q, top_k=1)
}

# Tolerance for matching timestamps (in seconds)
timestamp_tolerance = 10

# Store overall results
results = []

for method_name, retrieval_function in retrieval_methods.items():
    print(f"\nEvaluating {method_name}...")
    
    correct_answers = 0
    total_answerable = 0
    false_positives = 0
    total_unanswerable = 0
    latencies = []

    for idx, row in questions.iterrows():
        question = row["Question"]
        ground_truth = row["Timestamp (sec)"]
        is_answerable = row["Question Type"] == "Answerable"

        start_time = time.time()
        try:
            retrieved = retrieval_function(question)
        except Exception as e:
            print(f"Error during retrieval for question '{question}': {e}")
            retrieved = []
        end_time = time.time()

        latency = end_time - start_time
        latencies.append(latency)

        if is_answerable:
            total_answerable += 1
            if retrieved:
                retrieved_timestamp = retrieved[0]['timestamp']
                if abs(retrieved_timestamp - ground_truth) <= timestamp_tolerance:
                    correct_answers += 1
        else:
            total_unanswerable += 1
            if retrieved:
                false_positives += 1

    accuracy = correct_answers / total_answerable if total_answerable > 0 else 0
    rejection_quality = 1 - (false_positives / total_unanswerable) if total_unanswerable > 0 else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    results.append({
        "Retrieval Method": method_name,
        "Accuracy on Answerable Questions": round(accuracy, 3),
        "Rejection Quality on Unanswerable Questions": round(rejection_quality, 3),
        "Average Latency (seconds)": round(avg_latency, 3)
    })

# Create results dataframe
results_df = pd.DataFrame(results)

# Save results
results_df.to_csv('evaluation_results.csv', index=False)

print("\nEvaluation Complete! Results:")
print(results_df)
