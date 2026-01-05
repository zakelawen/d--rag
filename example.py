# -*- coding: utf-8 -*-
"""
run_conditional_acd_v3_combo_alpha.py
按 (alpha_pos, alpha_neg) 组合输出：每个组合一个 jsonl 文件。
策略规则：
- needs_retrieval = 0 -> noctx-normal（无上下文，贪婪）
- needs_retrieval = 1:
    d < HARMFUL_THRESHOLD      -> ctx-neg-acd（用 alpha_neg）
    d > USEFUL_THRESHOLD       -> ctx-pos-acd（用 alpha_pos）
    介于两者之间/None          -> ctx-normal（带上下文，贪婪）
"""

import os
import json
import time
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, GenerationConfig
from gen_wrapper import LMWrapper
from tqdm import tqdm

# ====== 配置 ======
MODEL_NAME = "/mnt/zhangjinshuo/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
CACHE_DIR = "./cache"
OFFLOAD_DIR = "./offload"
DEVICE_MAP = "auto"
MAX_MEMORY = None

INPUT_JSONL = "/home/qluai/zjs/TruthTorchLM-main/medmcqa/对比解码部分/my_method/ACD/qwen_data/medmcqa_seper_results_with_retrieval.jsonl"

HARMFUL_THRESHOLD = -0.1
USEFUL_THRESHOLD  =  0.3  # 确保 HARMFUL < USEFUL

MAX_NEW_TOKENS = 20
DO_SAMPLE = False  # 强制贪婪
GEN_BUDGET_TOKENS = 256
EVIDENCE_SEP = "\n\n---\n\n"

# 组合 α 列表
ALPHA_NEG_VALUES = [0.05, 0.1, 0.2]
ALPHA_POS_VALUES = [0.1, 0.5, 1.0]

# 输出前缀
BASE_OUTPUT_PATH = "/home/qluai/zjs/TruthTorchLM-main/medmcqa/对比解码部分/my_method/ACD/my_method_qwen_results/medmcqa"

# ====== 工具函数 ======
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

def build_stem(item: Dict[str, Any]) -> str:
    rq = item.get("raw_question")
    if rq and isinstance(rq, str) and rq.strip():
        stem = rq.strip()
        if not stem.lower().strip().endswith("answer:"):
            stem += "\nAnswer:"
        return stem
    q = (item.get("question") or "").strip()
    if not q:
        q = "Answer the multiple-choice question."
    if not q.lower().strip().endswith("answer:"):
        q += "\nAnswer:"
    return q

def token_len(tokenizer: AutoTokenizer, text: str) -> int:
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.size(1)

def build_pos_assistant_text(docs: List[Dict[str, Any]]) -> str:
    if not docs: return "No retrieved documents were provided."
    chunks = [f"[{i+1}] Title: {d.get('title') or f'doc_{i}'}\n{d.get('text') or ''}" for i, d in enumerate(docs)]
    body = EVIDENCE_SEP.join(chunks)
    prefix = ("The following retrieved documents are provided verbatim before the question. "
              "Answer the upcoming multiple-choice question with only the final option letter "
              "(e.g., A, B, C, D, or E) without explanations.")
    return prefix + "\n\n" + body

def build_neg_assistant_text(docs: List[Dict[str, Any]]) -> str:
    if not docs: return "No negative documents were provided."
    chunks = [f"[NEG-{i+1}] Title: {d.get('title') or f'doc_{i}'}\n{d.get('text') or ''}" for i, d in enumerate(docs)]
    body = EVIDENCE_SEP.join(chunks)
    prefix = ("The following retrieved documents are provided verbatim before the question. "
              "Answer the upcoming multiple-choice question with only the final option letter "
              "(e.g., A, B, C, D, or E) without explanations.")
    return prefix + "\n\n" + body

def apply_chat(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def ensure_dir_for_file(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def make_gen_config(tokenizer: AutoTokenizer,
                    do_sample: bool,
                    acd_alpha_pos: Optional[float]=None,
                    acd_alpha_neg: Optional[float]=None) -> GenerationConfig:
    kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,  # False: 贪婪
        return_dict_in_generate=True,
        output_scores=False,
    )
    if acd_alpha_pos is not None:
        kwargs["acd_alpha_pos"] = float(acd_alpha_pos)
    if acd_alpha_neg is not None:
        kwargs["acd_alpha_neg"] = float(acd_alpha_neg)
    return GenerationConfig(**kwargs)

def choose_strategy(needs_retrieval: int, dscore: Optional[float]) -> str:
    if not needs_retrieval:
        return "noctx-normal"
    if dscore is None:
        return "ctx-normal"
    if dscore < HARMFUL_THRESHOLD:
        return "ctx-neg-acd"
    if dscore > USEFUL_THRESHOLD:
        return "ctx-pos-acd"
    return "ctx-normal"

def build_main_prompt(tokenizer: AutoTokenizer, stem: str) -> str:
    main_messages = [
        {"role": "system", "content": "You are a helpful assistant. Your answer must be only the letter of the correct option (e.g., A, B, C, D, or E)."},
        {"role": "user", "content": stem},
    ]
    return apply_chat(tokenizer, main_messages)

def build_ctx_prompt(tokenizer: AutoTokenizer, stem: str, docs: List[Dict[str, Any]], polarity: str) -> (str, str):
    if polarity == "pos":
        ctx_text = build_pos_assistant_text(docs)
    else:
        ctx_text = build_neg_assistant_text(docs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": ctx_text},
        {"role": "user", "content": stem},
    ]
    return apply_chat(tokenizer, messages), ctx_text[:1200]

# ====== 单个组合的完整运行 ======
def run_single_combo(alpha_pos: float, alpha_neg: float,
                     base_output_path: str,
                     dataset: List[Dict[str, Any]],
                     tokenizer: AutoTokenizer,
                     model: LMWrapper):
    out_path = f"{base_output_path}_pos_{alpha_pos}_neg_{alpha_neg}.jsonl"
    ensure_dir_for_file(out_path)
    print(f"\n--- Running combo: POS={alpha_pos} | NEG={alpha_neg} ---")
    print(f"Output -> {out_path}")

    with open(out_path, "w", encoding="utf-8") as out_f:
        pbar = tqdm(enumerate(dataset), total=len(dataset), desc=f"pos={alpha_pos},neg={alpha_neg}", unit="sample")
        t0 = time.time()
        for idx, item in pbar:
            stem = build_stem(item)
            docs = item.get("docs") or []
            needs_retrieval = int(item.get("needs_retrieval", 1))
            dscore = item.get("d_seper_full", None)

            # 构主 prompt
            prompt_main = build_main_prompt(tokenizer, stem)
            # 可选长度提示（不强制）
            def _check_len(text: str, tag: str):
                L = token_len(tokenizer, text)
                model_max = getattr(tokenizer, "model_max_length", 32768)
                hard_limit = max(512, model_max - GEN_BUDGET_TOKENS)
                if L > hard_limit:
                    pbar.write(f"[Warn] #{idx} {tag} tokens={L} > {hard_limit} (model≈{model_max}).")
                return L
            main_len = _check_len(prompt_main, "main")

            strategy = choose_strategy(needs_retrieval, dscore)

            # 公共元信息
            meta = {
                "index": idx,
                "question": item.get("question"),
                "raw_question": item.get("raw_question"),
                "ground_truths": item.get("ground_truths"),
                "needs_retrieval": needs_retrieval,
                "d_seper_full_score": dscore,
                "doc_titles": [d.get("title") for d in docs] if docs else [],
                # 记录当前组合（即使该样本策略没用到某个 α）
                "combo_alpha_pos": alpha_pos,
                "combo_alpha_neg": alpha_neg,
                "strategy": strategy,
                "main_token_len": main_len,
            }

            # 执行
            if strategy == "noctx-normal":
                inputs = tokenizer(prompt_main, return_tensors="pt").to(model.device).input_ids
                gen_cfg = make_gen_config(tokenizer, DO_SAMPLE)
                with torch.no_grad():
                    out = model.generate_contrast(inputs=inputs, generation_config=gen_cfg, mode=None)
                gen_text = tokenizer.decode(out.sequences[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                rec = dict(meta, decoding_strategy="noctx-normal",
                           acd_alpha_pos=None, acd_alpha_neg=None,
                           model_output_raw=gen_text)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elif strategy == "ctx-normal":
                prompt_ctx, evidence_head = build_ctx_prompt(tokenizer, stem, docs, "pos")  # 中性前缀=pos版本
                _ = _check_len(prompt_ctx, "ctx-normal")
                inputs = tokenizer(prompt_ctx, return_tensors="pt").to(model.device).input_ids
                gen_cfg = make_gen_config(tokenizer, DO_SAMPLE)
                with torch.no_grad():
                    out = model.generate_contrast(inputs=inputs, generation_config=gen_cfg, mode=None)
                gen_text = tokenizer.decode(out.sequences[0][inputs.shape[1]:], skip_special_tokens=True).strip()
                rec = dict(meta, decoding_strategy="ctx-normal",
                           acd_alpha_pos=None, acd_alpha_neg=None,
                           model_output_raw=gen_text,
                           strategy_details={"type":"ctx-normal","token_len": token_len(tokenizer, prompt_ctx),"evidence_head": evidence_head})
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elif strategy == "ctx-neg-acd":
                prompt_neg, evidence_head = build_ctx_prompt(tokenizer, stem, docs, "neg")
                batch = tokenizer([prompt_main, prompt_neg], return_tensors="pt", padding="longest").to(model.device)
                inputs_main, inputs_neg = batch.input_ids[0:1], batch.input_ids[1:2]
                input_len = inputs_main.shape[1]
                gen_cfg = make_gen_config(tokenizer, DO_SAMPLE, acd_alpha_neg=alpha_neg)
                with torch.no_grad():
                    out = model.generate_contrast(
                        inputs=inputs_main,
                        inputs_neg=inputs_neg,
                        generation_config=gen_cfg,
                        mode="acd",
                    )
                gen_text = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True).strip()
                rec = dict(meta, decoding_strategy="ctx-neg-acd",
                           acd_alpha_pos=None, acd_alpha_neg=alpha_neg,
                           model_output_raw=gen_text,
                           strategy_details={"type":"ctx-neg-acd","token_len": token_len(tokenizer, prompt_neg),"evidence_head": evidence_head})
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            elif strategy == "ctx-pos-acd":
                prompt_pos, evidence_head = build_ctx_prompt(tokenizer, stem, docs, "pos")
                batch = tokenizer([prompt_main, prompt_pos], return_tensors="pt", padding="longest").to(model.device)
                inputs_main, inputs_pos = batch.input_ids[0:1], batch.input_ids[1:2]
                input_len = inputs_main.shape[1]
                gen_cfg = make_gen_config(tokenizer, DO_SAMPLE, acd_alpha_pos=alpha_pos)
                with torch.no_grad():
                    out = model.generate_contrast(
                        inputs=inputs_main,
                        inputs_pos=inputs_pos,
                        generation_config=gen_cfg,
                        mode="acd",
                    )
                gen_text = tokenizer.decode(out.sequences[0][input_len:], skip_special_tokens=True).strip()
                rec = dict(meta, decoding_strategy="ctx-pos-acd",
                           acd_alpha_pos=alpha_pos, acd_alpha_neg=None,
                           model_output_raw=gen_text,
                           strategy_details={"type":"ctx-pos-acd","token_len": token_len(tokenizer, prompt_pos),"evidence_head": evidence_head})
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()
    print(f"Done combo pos={alpha_pos}, neg={alpha_neg} -> {out_path}")

# ====== 主入口：跑所有 (pos, neg) 组合 ======
def main():
    assert HARMFUL_THRESHOLD < USEFUL_THRESHOLD, "要求：HARMFUL_THRESHOLD < USEFUL_THRESHOLD"

    print("Loading model and tokenizer...")
    model = LMWrapper(
        model_name_or_path=MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map=DEVICE_MAP,
        max_memory=MAX_MEMORY,
        offload_folder=OFFLOAD_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Ready.")
    print("-" * 30)

    if not os.path.exists(INPUT_JSONL):
        print(f"Error: dataset file not found -> {INPUT_JSONL}")
        return
    dataset = load_jsonl(INPUT_JSONL)
    print(f"Loaded {len(dataset)} items.")
    print("-" * 30)

    for alpha_pos in ALPHA_POS_VALUES:
        for alpha_neg in ALPHA_NEG_VALUES:
            run_single_combo(alpha_pos, alpha_neg, BASE_OUTPUT_PATH, dataset, tokenizer, model)

if __name__ == "__main__":
    main()
