## 如何使用

### 1.配置环境

需要安装以下依赖：

**安装python包**：

```bash
# 最小依赖
pip install onnxruntime onnx numpy
# 可选：用于ONNX GPU推理
pip install onnxruntime-gpu  # 如果使用GPU版本

```

**安装系统依赖**：

```bash
# 音频处理
apt-get update
apt-get install -y ffmpeg  
```

**安装建议**：
1. **CPU环境**：只安装`onnxruntime`, `onnx`, `numpy`
2. **GPU环境**：使用`onnxruntime-gpu`替代`onnxruntime`

### 2.运行
```bash
cd /to/path

# 下载 base 模型
git clone https://www.modelscope.cn/manyeyes/DolphinAsr-base-int8-onnx.git

# 或者下载 small 模型
git clone https://www.modelscope.cn/manyeyes/DolphinAsr-small-int8-onnx.git

# 以 base 模型为例
cd DolphinAsr-base-int8-onnx

# 将 dolphin_onnx_inferencer.py 
# 拷贝到 DolphinAsr-base-int8-onnx 文件夹

# 运行推理
python dolphin_onnx_inferencer.py 
```

### 3.其他说明

这是一个简单的onnx模型用例，没有实现自动语种检测，所以在测试时，请手动指定与音频相匹配的语种和地区，如：lang_sym="zh", region_sym="CN"

以下作为参考：
```python
LANGUAGE_REGION_CODES = {
    "zh-CN": ("Chinese (Mandarin)", "中文(普通话)"),
    "zh-TW": ("Chinese (Taiwan)", "中文(台湾)"),
    "zh-WU": ("Chinese (Wuyu)", "中文(吴语)"),
    "zh-SICHUAN": ("Chinese (Sichuan)", "中文(四川话)"),
    "zh-SHANXI": ("Chinese (Shanxi)", "中文(山西话)"),
    "zh-ANHUI": ("Chinese (Anhui)", "中文(安徽话)"),
    "zh-TIANJIN": ("Chinese (Tianjin)", "中文(天津话)"),
    "zh-NINGXIA": ("Chinese (Ningxia)", "中文(宁夏话)"),
    "zh-SHAANXI": ("Chinese (Shanxi)", "中文(陕西话)"),
    "zh-HEBEI": ("Chinese (Hebei)", "中文(河北话)"),
    "zh-SHANDONG": ("Chinese (Shandong)", "中文(山东话)"),
    "zh-GUANGDONG": ("Chinese (Guangdong)", "中文(广东话)"),
    "zh-SHANGHAI": ("Chinese (Shanghai)", "中文(上海话)"),
    "zh-HUBEI": ("Chinese (Hubei)", "中文(湖北话)"),
    "zh-LIAONING": ("Chinese (Liaoning)", "中文(辽宁话)"),
    "zh-GANSU": ("Chinese (Gansu)", "中文(甘肃话)"),
    "zh-FUJIAN": ("Chinese (Fujian)", "中文(福建话)"),
    "zh-HUNAN": ("Chinese (Hunan)", "中文(湖南话)"),
    "zh-HENAN": ("Chinese (Henan)", "中文(河南话)"),
    "zh-YUNNAN": ("Chinese (Yunnan)", "中文(云南话)"),
    "zh-MINNAN": ("Chinese (Minnan)", "中文(闽南语)"),
    "zh-WENZHOU": ("Chinese (Wenzhou)", "中文(温州话)"),
    "ja-JP": ("Japanese", "日语"),
    "th-TH": ("Thai", "泰语"),
    "ru-RU": ("Russian", "俄语"),
    "ko-KR": ("Korean", "韩语"),
    "id-ID": ("Indonesian", "印度尼西亚语"),
    "vi-VN": ("Vietnamese", "越南语"),
    "ct-NULL": ("Yue (Chinese)", "粤语"),
    "ct-HK": ("Yue (Hongkong)", "粤语(香港)"),
    "ct-GZ": ("Yue (Guangdong)", "粤语(广东)"),
    "hi-IN": ("Hindi", "印地语"),
    "ur-IN": ("Urdu", "乌尔都语(印度)"),
    "ur-PK": ("Urdu (Islamic Republic of Pakistan)", "乌尔都语"),
    "ms-MY": ("Malay", "马来语"),
    "uz-UZ": ("Uzbek", "乌兹别克语"),
    "ar-MA": ("Arabic (Morocco)", "阿拉伯语(摩洛哥)"),
    "ar-GLA": ("Arabic", "阿拉伯语"),
    "ar-SA": ("Arabic (Saudi Arabia)", "阿拉伯语(沙特)"),
    "ar-EG": ("Arabic (Egypt)", "阿拉伯语(埃及)"),
    "ar-KW": ("Arabic (Kuwait)", "阿拉伯语(科威特)"),
    "ar-LY": ("Arabic (Libya)", "阿拉伯语(利比亚)"),
    "ar-JO": ("Arabic (Jordan)", "阿拉伯语(约旦)"),
    "ar-AE": ("Arabic (U.A.E.)", "阿拉伯语(阿联酋)"),
    "ar-LVT": ("Arabic (Levant)", "阿拉伯语(黎凡特)"),
    "fa-IR": ("Persian", "波斯语"),
    "bn-BD": ("Bengali", "孟加拉语"),
    "ta-SG": ("Tamil (Singaporean)", "泰米尔语(新加坡)"),
    "ta-LK": ("Tamil (Sri Lankan)", "泰米尔语(斯里兰卡)"),
    "ta-IN": ("Tamil (India)", "泰米尔语(印度)"),
    "ta-MY": ("Tamil (Malaysia)", "泰米尔语(马来西亚)"),
    "te-IN": ("Telugu", "泰卢固语"),
    "ug-NULL": ("Uighur", "维吾尔语"),
    "ug-CN": ("Uighur", "维吾尔语"),
    "gu-IN": ("Gujarati", "古吉拉特语"),
    "my-MM": ("Burmese", "缅甸语"),
    "tl-PH": ("Tagalog", "塔加洛语"),
    "kk-KZ": ("Kazakh", "哈萨克语"),
    "or-IN": ("Oriya / Odia", "奥里亚语"),
    "ne-NP": ("Nepali", "尼泊尔语"),
    "mn-MN": ("Mongolian", "蒙古语"),
    "km-KH": ("Khmer", "高棉语"),
    "jv-ID": ("Javanese", "爪哇语"),
    "lo-LA": ("Lao", "老挝语"),
    "si-LK": ("Sinhala", "僧伽罗语"),
    "fil-PH": ("Filipino", "菲律宾语"),
    "ps-AF": ("Pushto", "普什图语"),
    "pa-IN": ("Panjabi", "旁遮普语"),
    "kab-NULL": ("Kabyle", "卡拜尔语"),
    "ba-NULL": ("Bashkir", "巴什基尔语"),
    "ks-IN": ("Kashmiri", "克什米尔语"),
    "tg-TJ": ("Tajik", "塔吉克语"),
    "su-ID": ("Sundanese", "巽他语"),
    "mr-IN": ("Marathi", "马拉地语"),
    "ky-KG": ("Kirghiz", "吉尔吉斯语"),
    "az-AZ": ("Azerbaijani", "阿塞拜疆语"),
}
```