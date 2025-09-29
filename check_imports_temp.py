import sys
import os

try:
    import sentence_transformers
    print(f"✅ sentence_transformers найден: {sentence_transformers.__file__}")
except ImportError:
    print("❌ sentence_transformers НЕ найден")

try:
    import faiss
    print(f"✅ faiss найден: {faiss.__file__}")
except ImportError:
    print("❌ faiss НЕ найден")

print("\nПути Python (sys.path):")
for p in sys.path:
    print(p)














