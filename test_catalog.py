import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from catalog_loader import get_catalog, get_index

items = get_catalog()
idx = get_index()
print(f"Loaded {len(items)} items")

results = idx.search("Java developer senior backend Spring SQL", top_k=5)
for r in results:
    print(f"  - {r['name']} ({r['test_type']})")

print("\nSearch: contact center customer service")
results2 = idx.search("contact center customer service entry level", top_k=5)
for r in results2:
    print(f"  - {r['name']} ({r['test_type']})")

print("\nSearch: personality leadership executive")
results3 = idx.search("personality leadership executive senior", top_k=5)
for r in results3:
    print(f"  - {r['name']} ({r['test_type']})")
