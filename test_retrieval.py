"""Quick retrieval validation — no LLM needed."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from catalog_loader import get_index

idx = get_index()

tests = [
    ("OPQ leadership executive CXO", ["Occupational Personality Questionnaire OPQ32r", "OPQ Leadership Report"]),
    ("contact center customer service entry level", ["Contact Center Call Simulation (New)", "Entry Level Customer Serv-Retail & Contact Center"]),
    ("numerical reasoning graduate financial analyst", ["SHL Verify Interactive - Numerical Reasoning"]),
    ("safety dependability manufacturing industrial", ["Dependability and Safety Instrument (DSI)"]),
    ("HIPAA medical terminology healthcare", ["HIPAA (Security)", "Medical Terminology (New)"]),
    ("Excel Word admin assistant simulation", ["MS Excel (New)", "MS Word (New)"]),
    ("Java Spring SQL AWS Docker backend engineer", ["Core Java (Advanced Level) (New)", "Spring (New)", "SQL (New)"]),
]

all_pass = True
for query, expected_names in tests:
    results = idx.search(query, top_k=10)
    result_names = [r["name"] for r in results]
    found = [n for n in expected_names if any(n.lower() in rn.lower() or rn.lower() in n.lower() for rn in result_names)]
    status = "PASS" if len(found) == len(expected_names) else "PARTIAL"
    if status != "PASS":
        all_pass = False
    print(f"[{status}] '{query[:50]}'")
    print(f"       Expected: {expected_names}")
    print(f"       Found:    {found}")
    print(f"       Top results: {result_names[:5]}")

print("\n" + ("ALL PASS" if all_pass else "SOME FAILURES"))
