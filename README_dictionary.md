# 关键词字典扩展模块

## 概述

本模块实现了关键词字典扩展功能，将关键词分为多个类别（confidence、hesitation、escalation、repair），并支持可配置的分数权重。

## 文件结构

```
all/
├── dictionary.py              # 核心模块
├── dictionary_config.yaml     # 配置文件
├── integration_example.py     # 集成示例
├── requirements.txt           # 依赖包（已更新）
└── README_dictionary.md       # 本文档
```

## 功能特性

### 1. 关键词分类

- **confidence** (信心): 分数增量 +2
- **hesitation** (犹豫): 分数增量 -2  
- **escalation** (升级): 分数增量 -3
- **repair** (修复): 分数增量 +1

### 2. 核心功能

- 加载YAML配置文件
- 关键词匹配和评分
- 支持正则表达式匹配
- 大小写不敏感匹配
- 批量处理
- 动态添加关键词

## 使用方法

### 基本使用

```python
from dictionary import DictionaryScorer

# 创建评分器
scorer = DictionaryScorer('dictionary_config.yaml')

# 评分单个文本
result = scorer.score_text("I think maybe this is correct")
print(result)
# 输出: {'category': 'hesitation', 'score_delta': -2, 'matched_keywords': ['I think maybe'], 'total_matches': 1}
```

### 批量处理

```python
# 批量评分
messages = [
    "I'm sure of it",
    "This is urgent!",
    "I'm sorry about that"
]

results = []
for message in messages:
    result = scorer.score_text(message)
    results.append(result)
```

### 集成到现有系统

```python
from integration_example import EnhancedChatScorer

# 创建增强的聊天评分器
enhanced_scorer = EnhancedChatScorer()

# 评分消息
result = enhanced_scorer.score_message_enhanced("I think maybe this is correct")
print(f"综合评分: {result['final_score']}")
```

## 配置文件格式

`dictionary_config.yaml` 文件格式：

```yaml
categories:
  confidence:
    score_delta: 2
    keywords:
      - "I'm sure of it"
      - "Yes, definitely"
      # ... 更多关键词
  hesitation:
    score_delta: -2
    keywords:
      - "I think maybe"
      - "I'm not certain"
      # ... 更多关键词
  # ... 其他类别

default_score: 0
case_sensitive: false
```

## 测试

运行测试：

```bash
# 测试基本功能
python dictionary.py

# 测试集成功能
python integration_example.py
```

## 示例输出

### 基本测试结果

```
=== 关键词字典评分测试 ===
配置文件路径: dictionary_config.yaml

可用类别:
  - confidence: 分数增量 2, 关键词数量 20
  - hesitation: 分数增量 -2, 关键词数量 21
  - escalation: 分数增量 -3, 关键词数量 20
  - repair: 分数增量 1, 关键词数量 20

测试结果:
1. 文本: 'I think maybe'
   预期类别: hesitation
   实际类别: hesitation
   分数增量: -2
   匹配关键词: ['I think maybe']
   状态: ✓
```

### 集成测试结果

```
消息: 'I think maybe this is correct'
基础评分 (原有逻辑): 60.0
字典评分 (新逻辑): 58.0
综合评分: 59.0
匹配类别: hesitation
分数增量: -2
匹配关键词: ['I think maybe']
```

## API 参考

### DictionaryScorer 类

#### 方法

- `__init__(config_path)`: 初始化评分器
- `score_text(text)`: 评分单个文本
- `get_categories()`: 获取所有类别
- `get_category_info(category_name)`: 获取类别信息
- `add_keyword(category_name, keyword)`: 添加关键词
- `save_config()`: 保存配置

#### 返回值格式

```python
{
    'category': 'hesitation',           # 匹配的类别
    'score_delta': -2,                  # 分数增量
    'matched_keywords': ['I think maybe'],  # 匹配的关键词
    'total_matches': 1                  # 总匹配次数
}
```

## 扩展和自定义

### 添加新类别

1. 编辑 `dictionary_config.yaml`
2. 添加新类别和关键词
3. 重新加载配置

### 调整分数权重

修改 `dictionary_config.yaml` 中的 `score_delta` 值。

### 添加关键词

```python
scorer = DictionaryScorer()
scorer.add_keyword('confidence', 'I am certain')
scorer.save_config()
```

## 依赖要求

- Python 3.7+
- PyYAML >= 6.0
- pandas >= 2.0.0
- numpy >= 1.20.0

安装依赖：

```bash
pip install -r requirements.txt
```

## 注意事项

1. 关键词匹配使用正则表达式，支持单词边界匹配
2. 默认大小写不敏感
3. 复合文本会匹配多个关键词，选择分数增量最大的类别
4. 配置文件使用UTF-8编码
5. 建议定期备份配置文件

## 故障排除

### 常见问题

1. **配置文件未找到**: 检查文件路径是否正确
2. **YAML格式错误**: 检查缩进和语法
3. **关键词不匹配**: 检查关键词是否在配置文件中
4. **编码问题**: 确保文件使用UTF-8编码

### 调试技巧

```python
# 启用详细输出
scorer = DictionaryScorer()
result = scorer.score_text("test message")
print(f"详细结果: {result}")

# 检查类别信息
for category in scorer.get_categories():
    info = scorer.get_category_info(category)
    print(f"{category}: {info}")
```
