import re
import os
import unicodedata
from typing import List, Dict, Set
from underthesea import word_tokenize

class TextPreprocessor:
    """
    Class xử lý tiền xử lý văn bản tiếng Việt & tiếng Anh cho bài toán Sentiment Analysis.
    """
    def __init__(self, dict_dir: str = None):
        if dict_dir is None:
            # Mặc định lấy thư mục dictionaries trong project
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dict_dir = os.path.join(os.path.dirname(current_dir), 'data', 'dictionaries')
        
        self.dict_dir = dict_dir
        self.stopwords = self._load_set_from_file(os.path.join(dict_dir, 'vietnamese-stopwords.txt'))
        self.teencode_dict = self._load_dict_from_file(os.path.join(dict_dir, 'teencode.txt'))
        self.wrong_words_dict = self._load_dict_from_file(os.path.join(dict_dir, 'wrong-word.txt'))
        self.emoji_dict = self._load_dict_from_file(os.path.join(dict_dir, 'emojicon.txt'))
        self.positive_emojis = self._load_set_from_file(os.path.join(dict_dir, 'positive_emoji.txt'))
        self.negative_emojis = self._load_set_from_file(os.path.join(dict_dir, 'negative_emoji.txt'))

    def _load_set_from_file(self, filepath: str) -> Set[str]:
        if not os.path.exists(filepath):
            return set()
        with open(filepath, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}

    def _load_dict_from_file(self, filepath: str) -> Dict[str, str]:
        if not os.path.exists(filepath):
            return {}
        result = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    result[parts[0].lower()] = parts[1].lower()
                elif len(parts) == 1 and ' ' in line:
                    subparts = line.strip().split(' ', 1)
                    if len(subparts) == 2:
                        result[subparts[0].lower()] = subparts[1].lower()
        return result

    def normalize_unicode(self, text: str) -> str:
        """Chuẩn hóa bảng mã Unicode (NFC)."""
        return unicodedata.normalize('NFC', text)

    def process_emojis(self, text: str) -> str:
        """Thay thế emoji/emojicon bằng từ mang sắc thái tích cực / tiêu cực."""
        for emo, replacement in self.emoji_dict.items():
            text = text.replace(emo, f" {replacement} ")
        for emo in self.positive_emojis:
            text = text.replace(emo, " tích cực ")
        for emo in self.negative_emojis:
            text = text.replace(emo, " tiêu cực ")
        return text

    def replace_teencode_and_typos(self, text: str) -> str:
        """Thay thế viết tắt, teencode, lỗi chính tả."""
        words = text.split()
        normalized_words = []
        for word in words:
            w_lower = word.lower()
            if w_lower in self.teencode_dict:
                normalized_words.append(self.teencode_dict[w_lower])
            elif w_lower in self.wrong_words_dict:
                normalized_words.append(self.wrong_words_dict[w_lower])
            else:
                normalized_words.append(word)
        return " ".join(normalized_words)

    def clean_text(self, text: str) -> str:
        """Làm sạch ký tự thừa, HTML tags, link, dấu câu không cần thiết."""
        if not isinstance(text, str):
            return ""
        
        text = self.normalize_unicode(text)
        text = self.process_emojis(text)
        
        # Xóa URL và email
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'\S+@\S+', ' ', text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Thay thế teencode & từ sai
        text = self.replace_teencode_and_typos(text)
        
        # Giữ lại các chữ cái tiếng Việt, số và khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        # Xóa khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize_vietnamese(self, text: str, remove_stopwords: bool = False) -> str:
        """Tách từ tiếng Việt bằng underthesea và tùy chọn loại bỏ stopwords."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return ""
        
        tokenized = word_tokenize(cleaned, format="text")
        
        if remove_stopwords:
            words = tokenized.split()
            words = [w for w in words if w not in self.stopwords]
            return " ".join(words)
        
        return tokenized
