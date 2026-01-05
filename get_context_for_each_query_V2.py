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

# ----------------------------------------------------------------
# 0. 日志 & 异步
# ----------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
nest_asyncio.apply()

# ----------------------------------------------------------------
# 1. 路径与输入
# ----------------------------------------------------------------
# 注意：保持这些路径与实际环境一致
storage_path = "/home/qluai/zjs/TruthTorchLM-main/medmcqa/对比解码部分/(prepare_work)llamaindex_RAG/storage"
file_paths = [
    "/home/qluai/zjs/MedMCQA数据处理/medmcqa/train_formatted_sample_5000.jsonl",
    "/home/qluai/zjs/TruthTorchLM-main/medqa/train_formatted_sample_5000.jsonl",
]

# 结果根目录：脚本所在目录下的 results/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
SUBDIR_MAP = {
    "medmcqa": os.path.join(RESULTS_DIR, "medmcqa"),
    "medqa": os.path.join(RESULTS_DIR, "medqa"),
}

# 创建子目录
for sub in SUBDIR_MAP.values():
    os.makedirs(sub, exist_ok=True)

# 启动前路径自检
to_check = [storage_path] + file_paths
missing = [p for p in to_check if not os.path.exists(p)]
if missing:
    sys.exit(f"[路径不存在] \n" + "\n".join(missing))

# ----------------------------------------------------------------
# 2. 模型初始化
# ----------------------------------------------------------------
# OpenAI（用于查询变体）。若失败，将自动降级为离线模式（num_queries=0）
print("Initializing query generation LLM (OpenAI gpt-4o-mini) if available...")
query_gen_llm = None
NUM_QUERIES = 2  # 默认启用查询变体
OPENAI_API_KEY = "sk-G8CJQUKCn17H6E3140C3166f4dF9447d9c8c4a1f56B77e66"  # 请在运行环境设置此变量
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.xty.app/v1")  # 如无代理可改回官方

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
    model_name="/mnt/zhangjinshuo/models--Qwen--Qwen3-Embedding-4B/snapshots/5cf2132abc99cad020ac570b19d031efec650f2b",
    model_kwargs={"local_files_only": True, "quantization_config": quantization_config_embed},
)
Settings.embed_model = embed_model
print("Embedding Model initialized.")

# 本地 LLM（备用；当前检索不强依赖）
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

# ----------------------------------------------------------------
# 3. 加载索引
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# 4. 初始化混合检索器
# ----------------------------------------------------------------
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

# ----------------------------------------------------------------
# 5. 批量执行检索：分别输出到 results/medmcqa 与 results/medqa
# ----------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n--- Starting batch retrieval for {len(file_paths)} file(s). ---")
print(f"--- Run timestamp: {timestamp} ---")
print(f"--- Results base dir: {RESULTS_DIR} ---")

def infer_subdir(path: str) -> str:
    """根据输入文件路径推断输出子目录：medmcqa 或 medqa"""
    p = path.lower()
    if "medmcqa" in p:
        return "medmcqa"
    if "medqa" in p:
        return "medqa"
    # 默认放到 results 根目录（不太可能用到）
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

                    # --- MODIFICATION START ---
                    # 根据您的要求，不再对问题进行清洗，直接使用原始问题进行检索
                    # cleaned_question = raw_question.split('\nA)')[0].strip()

                    # 检索 (使用原始问题)
                    nodes = retriever.retrieve(raw_question)
                    # --- MODIFICATION END ---


                    # 组织输出
                    formatted_docs = []
                    if nodes:
                        for nws in nodes:
                            node = nws.node
                            formatted_docs.append({
                                "title": getattr(node, "id_", None),
                                "text": getattr(node, "text", None),
                            })
                    
                    # --- MODIFICATION START ---
                    # 相应地，将结果中的 "question" 字段也设置为原始问题
                    result_for_one_question = {
                        "question": raw_question,
                        "raw_question": raw_question,
                        "docs": formatted_docs
                    }
                    # --- MODIFICATION END ---

                    outfile.write(json.dumps(result_for_one_question, ensure_ascii=False) + "\n")

                except json.JSONDecodeError:
                    # 跳过坏行
                    continue
                except Exception as e:
                    # 不让单条失败影响整体；可按需打开调试日志
                    # logging.exception(f"Line failed: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}. Skipping.")
        continue

print(f"\n✅ Batch processing complete.")
print(f"   medmcqa 输出目录: {SUBDIR_MAP['medmcqa']}")
print(f"   medqa   输出目录: {SUBDIR_MAP['medqa']}")
