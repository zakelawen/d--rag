import os
import sys
import json
import logging
from datetime import datetime
import torch
import faiss  # noqa: F401  # ensure faiss is importable
import nest_asyncio
from tqdm import tqdm

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from transformers import BitsAndBytesConfig


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
nest_asyncio.apply()



storage_path = "xxx"
file_paths = [
    "xxx",
    "xxx",
]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUBDIR_MAP = {
    "medmcqa": os.path.join(RESULTS_DIR, "medmcqa"),
    "medqa": os.path.join(RESULTS_DIR, "medqa"),
}


for sub in SUBDIR_MAP.values():
    os.makedirs(sub, exist_ok=True)


to_check = [storage_path] + file_paths
missing = [p for p in to_check if not os.path.exists(p)]
if missing:
    sys.exit(f"[路径不存在] \n" + "\n".join(missing))


print("Initializing query generation LLM (OpenAI gpt-4o-mini) if available...")
query_gen_llm = None
NUM_QUERIES = 2  
OPENAI_API_KEY = "sk-"  
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.xty.app/v1")  

if OPENAI_API_KEY:
    try:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        query_gen_llm = OpenAI(model="gpt-4o-mini", api_base=OPENAI_API_BASE)
        print("OpenAI LLM initialized successfully.")
    except Exception as e:
        print(f"[WARN] OpenAI init failed ({e}). Fallback to offline (no query expansion).")
        query_gen_llm = None
        NUM_QUERIES = 0
else:
    print("[INFO] OPENAI_API_KEY not set. Running offline (no query expansion).")
    NUM_QUERIES = 0

# 向量嵌入模型（本地）
print("Initializing Embedding Model (Qwen3-Embedding-4B)...")
quantization_config_embed = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
embed_model = HuggingFaceEmbedding(
    model_name="models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b",
    model_kwargs={"local_files_only": True, "quantization_config": quantization_config_embed},
)
Settings.embed_model = embed_model
print("Embedding Model initialized.")


print("Initializing local LLM (Llama-3.1-8B-Instruct)...")
quantization_config_llm = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
llm = HuggingFaceLLM(
    model_name="/mnt/zhangjinshuo/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/meta-llama3.1-8b",
    tokenizer_name="/mnt/zhangjinshuo/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/meta-llama3.1-8b",
    tokenizer_kwargs={"local_files_only": True},
    model_kwargs={"local_files_only": True, "quantization_config": quantization_config_llm},
)
Settings.llm = llm
print("Local LLM initialized.")


print("Loading index from storage...")
try:
    vector_store = FaissVectorStore.from_persist_dir(storage_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=storage_path
    )
    index = load_index_from_storage(storage_context=storage_context)
    print("Index loaded successfully.")
except Exception as e:
    print(f"Error loading index from {storage_path}: {e}")
    sys.exit(1)


print("Initializing QueryFusionRetriever...")
vector_retriever = index.as_retriever(similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=NUM_QUERIES,
    use_async=True,
    llm=query_gen_llm,  # 若为 None，则 num_queries 应为 0
)
print(f"QueryFusionRetriever initialized. num_queries={NUM_QUERIES}")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n--- Starting batch retrieval for {len(file_paths)} file(s). ---")
print(f"--- Run timestamp: {timestamp} ---")
print(f"--- Results base dir: {RESULTS_DIR} ---")

def infer_subdir(path: str) -> str:
    p = path.lower()
    if "medmcqa" in p:
        return "medmcqa"
    if "medqa" in p:
        return "medqa"
    return ""

for file_path in file_paths:
    subset = infer_subdir(file_path)
    if subset:
        out_dir = SUBDIR_MAP[subset]
        out_name = f"{subset}_{timestamp}.jsonl"  # 命名：medmcqa_<ts>.jsonl / medqa_<ts>.jsonl
    else:
        out_dir = RESULTS_DIR
        base = os.path.splitext(os.path.basename(file_path))[0]
        out_name = f"{base}_{timestamp}.jsonl"

    output_filename = os.path.join(out_dir, out_name)

    print(f"\nProcessing file: {file_path}")
    print(f" -> Output will be saved to: {output_filename}")

    try:
        # 计算总行数以精准进度条
        with open(file_path, 'r', encoding='utf-8') as f_for_count:
            num_lines = sum(1 for _ in f_for_count)

        with open(file_path, 'r', encoding='utf-8') as infile, \
             open(output_filename, "w", encoding="utf-8") as outfile:

            for line in tqdm(infile, total=num_lines, desc=f"Processing {subset or os.path.basename(file_path)}"):
                try:
                    data = json.loads(line)
                    raw_question = data.get('question')
                    if not raw_question:
                        continue

                    nodes = retriever.retrieve(raw_question)



                    formatted_docs = []
                    if nodes:
                        for nws in nodes:
                            node = nws.node
                            formatted_docs.append({
                                "title": getattr(node, "id_", None),
                                "text": getattr(node, "text", None),
                            })
                    

                    result_for_one_question = {
                        "question": raw_question,
                        "raw_question": raw_question,
                        "docs": formatted_docs
                    }

                    outfile.write(json.dumps(result_for_one_question, ensure_ascii=False) + "\n")

                except json.JSONDecodeError:

                    continue
                except Exception as e:
                    # logging.exception(f"Line failed: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}. Skipping.")
        continue

print(f"\n Batch processing complete.")
print(f"   medmcqa 输出目录: {SUBDIR_MAP['medmcqa']}")
print(f"   medqa   输出目录: {SUBDIR_MAP['medqa']}")
