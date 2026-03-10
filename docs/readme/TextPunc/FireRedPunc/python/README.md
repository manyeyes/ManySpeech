# FireRedPunc ONNX 标点恢复脚本使用说明

## 简介
基于 FireRedPunc ONNX 模型，对输入文本进行标点恢复（自动添加逗号、句号等），支持中英文混合。

## 安装依赖
```bash
pip install numpy onnxruntime transformers
```
如需 GPU 加速，用 `onnxruntime-gpu` 替换 `onnxruntime`。

## 准备模型文件
需要以下三个文件：
- **ONNX 模型**：如 `model.onnx`
- **词汇表**：`tokens.txt`（BERT 格式）
- **标点映射**：`out_dict`（每行一个标点，索引 0 为空格）

## 使用
1. **下载模型**：
    ```bash
    cd /to/path

    git clone https://modelscope.cn/models/manyeyes/FireRedPunc-zh-en-onnx.git

    # 进入模型目录
    cd FireRedPunc-zh-en-onnx

    # 将 fireredpunc_onnx_inference.py 
    # 拷贝到 FireRedPunc-zh-en-onnx 文件夹
    ```
2. **修改脚本配置**：将脚本开头的路径改为实际路径：
   ```python
   ONNX_PATH = "/path/to/model.onnx"
   VOCAB_PATH = "/path/to/tokens.txt"
   OUT_DICT_PATH = "/path/to/out_dict"
   ```
2. **运行测试**：
   ```bash
   python fireredpunc_onnx_inference.py
   ```
3. **代码调用**：
   ```python
   from fireredpunc_onnx_inference import inference
   result = inference("你好今天过得怎么样")
   print(result)  # 输出：你好，今天过得怎么样？
   ```

## 注意事项
- 标点映射文件索引必须与模型输出类别一致。
- 脚本会自动移除输入文本中已有的标点。
- 默认 CPU 推理，如需 GPU 修改 `providers=['CUDAExecutionProvider']`。