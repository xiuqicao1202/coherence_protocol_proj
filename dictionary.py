#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dictionary.py - Keyword dictionary scoring module

Features:
- Load keyword categories and score weights from a YAML config file
- Match and score input strings based on keywords
- Return the matched category and score delta

Usage:
    from dictionary import DictionaryScorer

    scorer = DictionaryScorer('dictionary_config.yaml')
    result = scorer.score_text("I think maybe this is correct")
    print(result)  # {'category': 'hesitation', 'score_delta': -2}
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


class DictionaryScorer:
    """
    Keyword dictionary scorer

    Scores text based on predefined keyword categories
    """

    def __init__(self, config_path: Union[str, Path] = "dictionary_config.yaml"):
        """
        Initialize the scorer

        Args:
            config_path: Path to the YAML config file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.categories = self._build_category_patterns()
        self.not_patterns = self._build_not_patterns()

    def _load_config(self) -> Dict:
        """Load YAML config file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"YAML config file format error: {e}")

    def _build_category_patterns(self) -> Dict[str, Dict]:
        """
        Build keyword matching patterns

        Returns:
            Dictionary containing patterns for each category
        """
        categories = {}
        case_sensitive = self.config.get('case_sensitive', False)

        for category_name, category_config in self.config.get('categories', {}).items():
            keywords = category_config.get('keywords', [])
            score_delta = category_config.get('score_delta', 0)

            # Build regex patterns
            patterns = []
            for keyword in keywords:
                # Escape special characters
                escaped_keyword = re.escape(keyword)
                # Create word boundary pattern
                pattern = r'\b' + escaped_keyword + r'\b'
                patterns.append(pattern)

            # Combine all patterns
            if patterns:
                combined_pattern = '|'.join(patterns)
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(combined_pattern, flags)
            else:
                compiled_pattern = None

            categories[category_name] = {
                'pattern': compiled_pattern,
                'score_delta': score_delta,
                'keywords': keywords
            }

        return categories

    def _build_not_patterns(self) -> Dict[str, any]:
        """
        Build not-pattern matching rules

        Returns:
            Dictionary containing not-pattern config
        """
        not_config = self.config.get('not_patterns', {})
        not_patterns = {}

        # Handle confidence_to_hesitation pattern
        confidence_to_hesitation = not_config.get('confidence_to_hesitation', [])
        if confidence_to_hesitation:
            patterns = []
            for phrase in confidence_to_hesitation:
                escaped_phrase = re.escape(phrase)
                pattern = r'\b' + escaped_phrase + r'\b'
                patterns.append(pattern)

            if patterns:
                combined_pattern = '|'.join(patterns)
                flags = 0 if self.config.get('case_sensitive', False) else re.IGNORECASE
                not_patterns['confidence_to_hesitation'] = {
                    'pattern': re.compile(combined_pattern, flags),
                    'phrases': confidence_to_hesitation,
                    'target_category': 'hesitation',
                    'score_delta': -2  # score delta for hesitation
                }

        # Handle generic not pattern
        if not_config.get('generic_not_pattern', False):
            # Match "not + any word"
            generic_pattern = r'\bnot\s+\w+\b'
            flags = 0 if self.config.get('case_sensitive', False) else re.IGNORECASE
            not_patterns['generic_not'] = {
                'pattern': re.compile(generic_pattern, flags),
                'target_category': 'hesitation',
                'score_delta': -1  # score delta for generic not pattern
            }

        return not_patterns

    def score_text(self, text: str) -> Dict[str, Union[str, int, List[str]]]:
        """
        Score text based on keywords

        Args:
            text: The text to score

        Returns:
            Dictionary with match results:
            {
                'category': matched category name,
                'score_delta': score delta,
                'matched_keywords': list of matched keywords,
                'total_matches': total number of matches
            }
            If no match, returns default values
        """
        if not text or not isinstance(text, str):
            return {
                'category': None,
                'score_delta': self.config.get('default_score', 0),
                'matched_keywords': [],
                'total_matches': 0
            }

        best_category = None
        best_score_delta = self.config.get('default_score', 0)
        all_matched_keywords = []
        total_matches = 0

        # Check not patterns first (higher priority)
        not_matches = self._check_not_patterns(text)
        if not_matches:
            all_matched_keywords.extend(not_matches['matched_keywords'])
            total_matches += not_matches['total_matches']

            # not patterns usually indicate hesitation, higher priority
            if abs(not_matches['score_delta']) > abs(best_score_delta):
                best_category = not_matches['category']
                best_score_delta = not_matches['score_delta']

        # Check regular category keywords
        for category_name, category_info in self.categories.items():
            pattern = category_info['pattern']
            if pattern is None:
                continue

            # Find all matches
            matches = pattern.findall(text)
            if matches:
                matched_keywords = list(set(matches))  # deduplicate
                all_matched_keywords.extend(matched_keywords)
                total_matches += len(matches)

                # Choose the category with the largest absolute score delta
                score_delta = category_info['score_delta']
                if abs(score_delta) > abs(best_score_delta):
                    best_category = category_name
                    best_score_delta = score_delta

        return {
            'category': best_category,
            'score_delta': best_score_delta,
            'matched_keywords': list(set(all_matched_keywords)),  # deduplicate
            'total_matches': total_matches
        }

    def _check_not_patterns(self, text: str) -> Optional[Dict[str, Union[str, int, List[str]]]]:
        """
        Check for not patterns in the text

        Args:
            text: The text to check

        Returns:
            If a not pattern is matched, returns the match result; otherwise returns None
        """
        best_not_category = None
        best_not_score = 0
        not_matched_keywords = []
        not_total_matches = 0

        for pattern_name, pattern_info in self.not_patterns.items():
            pattern = pattern_info['pattern']
            matches = pattern.findall(text)

            if matches:
                matched_keywords = list(set(matches))
                not_matched_keywords.extend(matched_keywords)
                not_total_matches += len(matches)

                # Choose the not pattern with the largest absolute score delta
                score_delta = pattern_info['score_delta']
                if abs(score_delta) > abs(best_not_score):
                    best_not_category = pattern_info['target_category']
                    best_not_score = score_delta

        if best_not_category:
            return {
                'category': best_not_category,
                'score_delta': best_not_score,
                'matched_keywords': not_matched_keywords,
                'total_matches': not_total_matches
            }

        return None

    def get_categories(self) -> List[str]:
        """Get all available category names"""
        return list(self.categories.keys())

    def get_category_info(self, category_name: str) -> Optional[Dict]:
        """Get information for a specific category"""
        return self.categories.get(category_name)

    def add_keyword(self, category_name: str, keyword: str) -> bool:
        """
        Add a keyword to a specific category

        Args:
            category_name: Category name
            keyword: Keyword to add

        Returns:
            Whether the addition was successful
        """
        if category_name not in self.categories:
            return False

        # Add to config
        if 'categories' not in self.config:
            self.config['categories'] = {}
        if category_name not in self.config['categories']:
            self.config['categories'][category_name] = {'keywords': [], 'score_delta': 0}

        if keyword not in self.config['categories'][category_name]['keywords']:
            self.config['categories'][category_name]['keywords'].append(keyword)

            # Rebuild patterns
            self.categories = self._build_category_patterns()
            self.not_patterns = self._build_not_patterns()
            return True

        return False

    def save_config(self) -> bool:
        """
        Save config to file

        Returns:
            Whether the save was successful
        """
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"Failed to save config file: {e}")
            return False


def test_with_sample_data():
    """
    Test functionality with provided sample data
    """
    # Sample data
    sample_data = [
        {"text": "I think maybe", "category": "hesitation"},
        {"text": "I'm sure of it", "category": "confidence"},
        {"text": "This is urgent!", "category": "escalation"},
        {"text": "I'm sorry about that", "category": "repair"},
        {"text": "I'm not certain", "category": "hesitation"},
        {"text": "Yes, definitely", "category": "confidence"},
        # Additional not pattern test cases
        {"text": "I'm not sure about this", "category": "hesitation"},
        {"text": "This is not clear to me", "category": "hesitation"},
        {"text": "I'm not confident in this solution", "category": "hesitation"},
        {"text": "This is not working properly", "category": "hesitation"},
        {"text": "I'm not ready for this", "category": "hesitation"}
    ]

    print("=== Keyword Dictionary Scoring Test ===")
    print(f"Config file path: dictionary_config.yaml")
    print()

    try:
        # Create scorer
        scorer = DictionaryScorer()

        # Show available categories
        print("Available categories:")
        for category in scorer.get_categories():
            info = scorer.get_category_info(category)
            print(f"  - {category}: score delta {info['score_delta']}, number of keywords {len(info['keywords'])}")
        print()

        # Test each sample
        print("Test results:")
        for i, sample in enumerate(sample_data, 1):
            text = sample['text']
            expected_category = sample['category']

            result = scorer.score_text(text)
            actual_category = result['category']
            score_delta = result['score_delta']
            matched_keywords = result['matched_keywords']

            status = "✓" if actual_category == expected_category else "✗"
            print(f"{i}. Text: '{text}'")
            print(f"   Expected category: {expected_category}")
            print(f"   Actual category: {actual_category}")
            print(f"   Score delta: {score_delta}")
            print(f"   Matched keywords: {matched_keywords}")
            print(f"   Status: {status}")
            print()

        # Test complex text
        print("Complex text test:")
        complex_text = "I think maybe this is urgent, but I'm sorry about the confusion"
        result = scorer.score_text(complex_text)
        print(f"Text: '{complex_text}'")
        print(f"Result: {result}")
        print()

        # Test not patterns
        print("Not pattern special test:")
        not_test_cases = [
            "I'm not sure",
            "This is not clear",
            "I'm not confident",
            "This is not working",
            "I'm not ready",
            "This is not good",
            "I'm not certain about this",
            "The system is not stable"
        ]

        for i, test_text in enumerate(not_test_cases, 1):
            result = scorer.score_text(test_text)
            print(f"{i}. Text: '{test_text}'")
            print(f"   Category: {result['category']}")
            print(f"   Score delta: {result['score_delta']}")
            print(f"   Matched keywords: {result['matched_keywords']}")
            print()

        print("Test complete!")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_with_sample_data()
