"""
Evaluation Dashboard

Streamlit-based dashboard for visualizing RAG evaluation results.

Features:
- Metrics overview (pass/fail rates, latency)
- Per-query result cards
- Modality distribution charts
- Export/run evaluation controls

Run with: streamlit run dashboard.py
"""

import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from evaluation.harness import EvaluationHarness


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="RAG Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    /* Modern dark theme overrides */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stMetric label {
        color: #a0a0a0 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
    }
    
    /* Success/failure badges */
    .pass-badge {
        background: linear-gradient(135deg, #00c853 0%, #00e676 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    .fail-badge {
        background: linear-gradient(135deg, #ff5252 0%, #ff1744 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    
    /* Card styling */
    .result-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def load_evaluation_results(results_path: str):
    """Load evaluation results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def run_evaluation(use_llm: bool = False):
    """Run fresh evaluation and return results."""
    harness = EvaluationHarness(
        vector_store_path="./outputs/vectordb",
        queries_path="./evaluation/test_queries.json"
    )
    
    # Override LLM setting
    harness.generator.use_llm = use_llm
    if use_llm:
        try:
            from src.generation.llm_client import GeminiClient
            harness.generator.llm_client = GeminiClient()
            harness.generator._llm_available = True
        except Exception:
            harness.generator._llm_available = False
    
    harness.run()
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./evaluation/logs/results_{timestamp}.json"
    harness.export_results(output_path)
    
    return harness, output_path


def get_latest_results_file():
    """Find the most recent results file."""
    logs_dir = Path("./evaluation/logs")
    if not logs_dir.exists():
        return None
    
    result_files = list(logs_dir.glob("results_*.json"))
    if not result_files:
        return None
    
    return str(max(result_files, key=lambda p: p.stat().st_mtime))


def compute_metrics(results: dict):
    """Compute summary metrics from results."""
    if not results or 'results' not in results:
        return {}
    
    queries = results['results']
    total = len(queries)
    
    if total == 0:
        return {}
    
    # Pass/fail counts
    passed = sum(1 for q in queries if not q.get('failures'))
    failed = total - passed
    
    # Category breakdown
    categories = {}
    for q in queries:
        cat = q.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = {'total': 0, 'passed': 0}
        categories[cat]['total'] += 1
        if not q.get('failures'):
            categories[cat]['passed'] += 1
    
    # Latency stats
    retrieval_times = [q['retrieval']['time_ms'] for q in queries if 'retrieval' in q]
    generation_times = [q['answer']['generation_time_ms'] for q in queries if 'answer' in q]
    
    avg_retrieval = sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0
    avg_generation = sum(generation_times) / len(generation_times) if generation_times else 0
    
    # Modality usage
    modality_counts = {}
    for q in queries:
        if 'retrieval' in q:
            dist = q['retrieval'].get('modality_distribution', {})
            for mod, count in dist.items():
                modality_counts[mod] = modality_counts.get(mod, 0) + count
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / total * 100,
        'categories': categories,
        'avg_retrieval_ms': avg_retrieval,
        'avg_generation_ms': avg_generation,
        'modality_counts': modality_counts,
    }


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    
    # Run evaluation button
    st.markdown("### Run Evaluation")
    use_llm = st.checkbox("Use LLM for generation", value=True)
    
    if st.button("🚀 Run Fresh Evaluation", type="primary", use_container_width=True):
        with st.spinner("Running evaluation..."):
            harness, output_path = run_evaluation(use_llm=use_llm)
            st.success(f"Evaluation complete! Results saved to:\n{output_path}")
            st.cache_data.clear()
    
    st.divider()
    
    # Results file selection
    st.markdown("### Load Results")
    
    logs_dir = Path("./evaluation/logs")
    if logs_dir.exists():
        result_files = sorted(logs_dir.glob("results_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        file_options = [f.name for f in result_files[:10]]  # Last 10 runs
        
        if file_options:
            selected_file = st.selectbox("Select results file:", file_options)
            results_path = str(logs_dir / selected_file)
        else:
            st.info("No evaluation results found. Run an evaluation first.")
            results_path = None
    else:
        st.info("Evaluation logs directory not found.")
        results_path = None
    
    st.divider()
    
    # Document info
    st.markdown("### 📄 Document Info")
    st.markdown("""
    **Document:** Qatar Article IV 2024  
    **Format:** PDF (Multi-modal)  
    **Modalities:** Text, Tables, Figures
    """)


# =============================================================================
# MAIN CONTENT
# =============================================================================

# Header
st.markdown('<h1 class="dashboard-header">📊 RAG Evaluation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Comprehensive evaluation metrics for the Multi-Modal RAG system")

st.divider()

# Load results
if results_path:
    results = load_evaluation_results(results_path)
    
    if results:
        metrics = compute_metrics(results)
        
        # =============================================================================
        # METRICS OVERVIEW
        # =============================================================================
        
        st.markdown("## 📈 Metrics Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="Total Queries",
                value=metrics.get('total', 0)
            )
        
        with col2:
            st.metric(
                label="Passed",
                value=metrics.get('passed', 0),
                delta=f"{metrics.get('pass_rate', 0):.0f}%"
            )
        
        with col3:
            st.metric(
                label="Failed",
                value=metrics.get('failed', 0),
                delta=f"-{metrics.get('failed', 0)}" if metrics.get('failed', 0) > 0 else None,
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                label="Avg Retrieval",
                value=f"{metrics.get('avg_retrieval_ms', 0):.0f}ms"
            )
        
        with col5:
            st.metric(
                label="Avg Generation",
                value=f"{metrics.get('avg_generation_ms', 0):.0f}ms"
            )
        
        st.divider()
        
        # =============================================================================
        # CHARTS
        # =============================================================================
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 📊 Pass Rate by Category")
            
            categories = metrics.get('categories', {})
            if categories:
                import pandas as pd
                
                cat_data = []
                for cat, stats in categories.items():
                    cat_data.append({
                        'Category': cat.replace('_', ' ').title(),
                        'Passed': stats['passed'],
                        'Failed': stats['total'] - stats['passed'],
                    })
                
                df = pd.DataFrame(cat_data)
                st.bar_chart(df.set_index('Category'))
            else:
                st.info("No category data available")
        
        with col_right:
            st.markdown("### 🏷️ Modality Distribution")
            
            modality_counts = metrics.get('modality_counts', {})
            if modality_counts:
                import pandas as pd
                
                mod_data = pd.DataFrame([
                    {'Modality': k, 'Count': v} 
                    for k, v in modality_counts.items()
                ])
                st.bar_chart(mod_data.set_index('Modality'))
            else:
                st.info("No modality data available")
        
        st.divider()
        
        # =============================================================================
        # PER-QUERY RESULTS
        # =============================================================================
        
        st.markdown("## 📝 Per-Query Results")
        
        for query_result in results.get('results', []):
            query_id = query_result.get('query_id', 'Unknown')
            query_text = query_result.get('query', '')
            failures = query_result.get('failures', [])
            category = query_result.get('category', 'unknown')
            
            # Status badge
            status_badge = "✅ PASS" if not failures else "❌ FAIL"
            status_color = "green" if not failures else "red"
            
            with st.expander(f"{status_badge} | **{query_id}** - {query_text[:60]}{'...' if len(query_text) > 60 else ''}"):
                
                # Query info row
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Category:** `{category}`")
                
                with col2:
                    if 'retrieval' in query_result:
                        rt = query_result['retrieval'].get('time_ms', 0)
                        st.markdown(f"**Retrieval Time:** {rt:.0f}ms")
                
                with col3:
                    if 'answer' in query_result:
                        gt = query_result['answer'].get('generation_time_ms', 0)
                        st.markdown(f"**Generation Time:** {gt:.0f}ms")
                
                # Full query
                st.markdown(f"**Query:** {query_text}")
                
                # Answer
                if 'answer' in query_result:
                    answer = query_result['answer']
                    st.markdown("**Answer:**")
                    st.markdown(f"> {answer.get('answer_text', 'N/A')[:500]}")
                    
                    conf = answer.get('confidence', 'unknown')
                    ans_type = answer.get('answer_type', 'unknown')
                    st.markdown(f"**Type:** `{ans_type}` | **Confidence:** `{conf}`")
                
                # Retrieval info
                if 'retrieval' in query_result:
                    retrieval = query_result['retrieval']
                    mod_dist = retrieval.get('modality_distribution', {})
                    st.markdown(f"**Retrieved Modalities:** {mod_dist}")
                
                # Failures
                if failures:
                    st.error(f"**Failures:** {', '.join(failures)}")
                
                # Citations
                if 'answer' in query_result and query_result['answer'].get('citations'):
                    st.markdown("**Citations:**")
                    for cit in query_result['answer']['citations'][:3]:
                        page = cit.get('page_number', '?')
                        mod = cit.get('modality', '?')
                        st.markdown(f"- Page {page} ({mod})")
        
        st.divider()
        
        # =============================================================================
        # RAW DATA
        # =============================================================================
        
        with st.expander("📄 View Raw JSON Data"):
            st.json(results)
    
    else:
        st.warning("Could not load evaluation results. The file may be corrupted.")

else:
    # No results available - show instructions
    st.info("""
    ### 👋 Welcome to the RAG Evaluation Dashboard!
    
    No evaluation results found yet. To get started:
    
    1. Click **"🚀 Run Fresh Evaluation"** in the sidebar to run a new evaluation
    2. Or run the evaluation harness manually:
       ```bash
       python evaluation/harness.py
       ```
    3. The results will appear here automatically
    """)
    
    # Show expected file structure
    with st.expander("📁 Expected File Structure"):
        st.code("""
evaluation/
├── harness.py           # Evaluation harness
├── test_queries.json    # Test query definitions
└── logs/
    └── results_*.json   # Evaluation results
        """)


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col2:
    st.markdown(
        "<p style='text-align: center; color: #888;'>"
        "Multi-Modal RAG System • Evaluation Dashboard"
        "</p>",
        unsafe_allow_html=True
    )
