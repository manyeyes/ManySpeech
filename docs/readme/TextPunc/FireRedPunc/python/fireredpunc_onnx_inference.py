# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 manyeyes
Description: FireRedPunc ONNX 标点恢复推理脚本。
Date:       2025-02-14
"""
import re
import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer

# ========== 配置路径（请根据实际情况修改）==========
ONNX_PATH = "/path/to/model.onnx"
VOCAB_PATH = "/path/to/tokens.txt"
OUT_DICT_PATH = "/path/to/out_dict"
# ==============================================

# 加载标点映射文件
with open(OUT_DICT_PATH, 'r', encoding='utf-8') as f:
    out_lines = [line.strip() for line in f.readlines() if line.strip() != '']
punc_map = {i: sym for i, sym in enumerate(out_lines)}

# 加载 tokenizer
tokenizer = BertTokenizer(vocab_file=VOCAB_PATH, do_lower_case=True)

# 加载 ONNX 模型（CPU 推理）
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])


def remove_punc_and_fix_space(text):
    """移除文本中已有的标点，仅保留中文字符"""
    text = re.sub("[，。？！,\.?!]", " ", text)
    pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f])')
    parts = pattern.split(text.strip())
    parts = [p for p in parts if p.strip()]
    return "".join(parts)


def add_punc_to_text(tokens, preds):
    """
    根据分词结果和预测的标点类别索引，重建带标点的文本
    """
    txt = ""
    for i, (token, p) in enumerate(zip(tokens, preds)):
        # 处理 BERT 子词（如 "##ing"）
        if token.startswith("##"):
            token = token[2:]
        # 英文或数字前添加空格
        elif re.search("[a-zA-Z0-9#]+", token) and i > 0 and re.search("[a-zA-Z0-9#]+", tokens[i-1]):
            if preds[i-1] == 0:  # 前一个 token 后无标点
                txt += " "
        txt += token
        # 添加标点（非空格类别）
        if p != 0:
            txt += punc_map.get(p, '').split()[0]
    # 合并多余空格
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def inference(text):
    """对输入文本执行标点恢复"""
    cleaned = remove_punc_and_fix_space(text)
    if not cleaned:
        return ""

    tokens = tokenizer.tokenize(cleaned)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids_np = np.array([input_ids], dtype=np.int64)
    lengths_np = np.array([len(input_ids)], dtype=np.int64)

    # ONNX 推理
    outputs = session.run(None, {'input_ids': input_ids_np, 'lengths': lengths_np})
    logits = outputs[0]                     # shape: (1, seq_len, num_classes)
    preds = np.argmax(logits, axis=-1)[0]    # shape: (seq_len,)

    return add_punc_to_text(tokens, preds)


if __name__ == "__main__":
    test_text = "今天天气真不错我们出去散步吧The weather is really nice today. Let's go out for a walk"
    print(f"输入: {test_text}")
    result = inference(test_text)
    print(f"输出: {result}")