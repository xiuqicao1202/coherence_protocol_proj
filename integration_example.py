#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integration_example.py - 集成示例

展示如何将dictionary.py模块集成到现有的评分引擎中
"""

from dictionary import DictionaryScorer
import pandas as pd
import numpy as np


class EnhancedChatScorer:
    """
    增强的聊天评分器，集成了关键词字典评分功能
    """
    
    def __init__(self, dictionary_config_path="dictionary_config.yaml"):
        """
        初始化增强的聊天评分器
        
        Args:
            dictionary_config_path: 关键词字典配置文件路径
        """
        self.dictionary_scorer = DictionaryScorer(dictionary_config_path)
        
        # 原有的关键词字典（保持向后兼容）
        self.stress_words = {
            "error", "fail", "failed", "issue", "stuck", "delay", "urgent", "can't", "cannot", "wtf", "broken", "retry", "oops", "panic",
            "problem", "trouble", "difficult", "hard", "slow", "blocked", "miss", "lost", "down", "crash", "freeze", "hang", "stop",
            "unavailable", "unresponsive", "disconnect", "timeout", "overload", "overheat", "warning", "alert", "critical", "fatal",
            "danger", "risk", "unstable", "unexpected", "abnormal", "corrupt", "invalid", "denied", "refused", "rejected", "conflict",
            "collision", "late", "postpone", "stress", "tired", "exhausted", "annoy", "frustrate", "angry", "upset", "disappointed",
            "sad", "complain", "regret", "sorry", "pain", "hurt", "worry", "afraid", "fear", "scared", "terrified", "confuse", "mess",
            "chaos", "disaster", "unlucky", "unfortunate", "unable", "incomplete", "unhappy", "hopeless", "helpless", "useless",
            "pointless", "meaningless", "worthless", "bug", "glitch", "lag", "dead", "jam", "sucks", "hate", "dislike", "disgust"
        }
        self.positive_words = {
            "ok", "done", "ready", "roger", "nice", "great", "thanks", "thank you", "proceed", "confirm", "confirmed", "noted", "on it",
            "looks fine", "received", "good", "well done", "perfect", "excellent", "awesome", "fine", "clear", "all good", "smooth",
            "success", "successful", "fixed", "solved", "resolved", "stable", "fast", "faster", "quick", "quickly", "improve", "improved",
            "improving", "improvement", "safe", "safely", "safety", "reliable", "reliably", "trust", "trustworthy", "cooperate",
            "cooperation", "help", "helpful", "productive", "motivated", "enthusiastic", "engaged", "committed", "willing", "coordinated",
            "organized", "structured", "planned", "prepared", "ready", "available", "responsive", "usable", "workable", "manageable",
            "controllable", "predictable", "support", "supported", "supporting", "supportive", "cheer", "cheers", "happy", "happiness",
            "enjoy", "enjoyed", "enjoying", "enjoys", "like", "liked", "liking", "likes", "love", "loved", "loving", "loves"
        }
    
    def score_message_enhanced(self, message: str) -> dict:
        """
        增强的消息评分函数，结合原有逻辑和新的关键词字典
        
        Args:
            message: 要评分的消息
            
        Returns:
            包含详细评分信息的字典
        """
        if message is None or message == "nan":
            return {
                'base_score': np.nan,
                'dictionary_score': np.nan,
                'final_score': np.nan,
                'dictionary_result': None,
                'legacy_result': None
            }
        
        # 使用新的关键词字典评分
        dictionary_result = self.dictionary_scorer.score_text(message)
        
        # 使用原有的关键词评分逻辑
        message_lc = str(message).lower()
        s_cnt = sum(1 for w in self.stress_words if w in message_lc)
        p_cnt = sum(1 for w in self.positive_words if w in message_lc)
        
        # 原有评分逻辑
        base_score = 60.0
        base_score += 8.0 * p_cnt
        base_score -= 10.0 * s_cnt
        base_score = float(np.clip(base_score, 0, 100))
        
        # 新的关键词字典评分
        dictionary_score_delta = dictionary_result['score_delta']
        dictionary_score = 60.0 + dictionary_score_delta
        dictionary_score = float(np.clip(dictionary_score, 0, 100))
        
        # 综合评分（可以调整权重）
        # 这里使用简单的平均，实际应用中可以根据需要调整
        final_score = (base_score + dictionary_score) / 2.0
        final_score = float(np.clip(final_score, 0, 100))
        
        return {
            'base_score': base_score,
            'dictionary_score': dictionary_score,
            'final_score': final_score,
            'dictionary_result': dictionary_result,
            'legacy_result': {
                'stress_count': s_cnt,
                'positive_count': p_cnt,
                'stress_words_found': [w for w in self.stress_words if w in message_lc],
                'positive_words_found': [w for w in self.positive_words if w in message_lc]
            }
        }
    
    def score_messages_batch(self, messages: list) -> pd.DataFrame:
        """
        批量评分消息
        
        Args:
            messages: 消息列表
            
        Returns:
            包含评分结果的DataFrame
        """
        results = []
        for i, message in enumerate(messages):
            result = self.score_message_enhanced(message)
            result['message_id'] = i
            result['message'] = message
            results.append(result)
        
        return pd.DataFrame(results)


def demo_integration():
    """
    演示集成功能
    """
    print("=== 增强聊天评分器集成演示 ===")
    print()
    
    # 创建增强的聊天评分器
    scorer = EnhancedChatScorer()
    
    # 测试消息
    test_messages = [
        "I think maybe this is correct",  # hesitation
        "I'm sure of it, this will work",  # confidence
        "This is urgent! We need to fix it now!",  # escalation
        "I'm sorry about that mistake",  # repair
        "Great job! Thanks for your help",  # positive
        "This is broken and causing problems",  # stress
        "I'm not certain about this approach, but I think maybe we should try it",  # hesitation + confidence
        "I'm sorry about the confusion, but this is urgent and needs immediate attention"  # repair + escalation
    ]
    
    print("测试消息评分结果:")
    print("=" * 80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{i}. 消息: '{message}'")
        result = scorer.score_message_enhanced(message)
        
        print(f"   基础评分 (原有逻辑): {result['base_score']:.1f}")
        print(f"   字典评分 (新逻辑): {result['dictionary_score']:.1f}")
        print(f"   综合评分: {result['final_score']:.1f}")
        
        if result['dictionary_result']['category']:
            print(f"   匹配类别: {result['dictionary_result']['category']}")
            print(f"   分数增量: {result['dictionary_result']['score_delta']}")
            print(f"   匹配关键词: {result['dictionary_result']['matched_keywords']}")
        
        if result['legacy_result']['stress_words_found']:
            print(f"   原有压力词: {result['legacy_result']['stress_words_found']}")
        if result['legacy_result']['positive_words_found']:
            print(f"   原有积极词: {result['legacy_result']['positive_words_found']}")
    
    print("\n" + "=" * 80)
    print("批量评分结果:")
    
    # 批量评分
    batch_results = scorer.score_messages_batch(test_messages)
    
    # 显示关键列
    display_cols = ['message', 'base_score', 'dictionary_score', 'final_score']
    print(batch_results[display_cols].to_string(index=False))
    
    print("\n集成演示完成!")


if __name__ == "__main__":
    demo_integration()
