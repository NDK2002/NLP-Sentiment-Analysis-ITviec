import os
import joblib
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class FeatureExtractor:
    """
    Class quản lý việc vector hóa dữ liệu văn bản bằng BoW hoặc TF-IDF.
    """
    def __init__(self, method: str = 'tfidf', max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True
            )
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range
            )
        else:
            raise ValueError(f"Phương thức '{method}' không được hỗ trợ. Dùng 'tfidf' hoặc 'bow'.")

    def fit_transform(self, corpus: List[str]):
        """Học từ điển và vector hóa tập dữ liệu."""
        return self.vectorizer.fit_transform(corpus)

    def transform(self, corpus: List[str]):
        """Vector hóa tập dữ liệu dựa trên từ điển đã học."""
        return self.vectorizer.transform(corpus)

    def save(self, filepath: str):
        """Lưu vectorizer đã fit vào file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        print(f"Đã lưu vectorizer tại: {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Tải vectorizer từ file."""
        instance = cls()
        instance.vectorizer = joblib.load(filepath)
        return instance
