"""Regression test for generation fixes."""

import sys
sys.path.insert(0, ".")

from demo import RAGDemo


def test_query(demo, query, expected_type=None, expected_intent=None):
    """Run a query and report results."""
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("="*60)
    
    demo.process_query(query)


def main():
    demo = RAGDemo()
    
    # Test queries
    queries = [
        # TEXT_SUMMARY
        "Summarize the outlook and risks",
        "Explain Qatar's economic outlook",
        
        # POLICY_QUESTION  
        "What are the fiscal policy recommendations?",
        
        # TABLE_LOOKUP
        "Show me Table 1",
        
        # UNKNOWN - should be NOT_FOUND
        "What is the population of Qatar?",
        "What is the weather in Doha?",
    ]
    
    for q in queries:
        test_query(demo, q)


if __name__ == "__main__":
    main()
