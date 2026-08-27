import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import cross_val_score

class SentimentModelTrainer:
    """
    Quản lý huấn luyện, so sánh và đánh giá các mô hình phân loại cảm xúc.
    """
    def __init__(self):
        self.models: Dict[str, Any] = {
            "Multinomial Naive Bayes": MultinomialNB(),
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            "Support Vector Machine (Linear)": SVC(kernel='linear', class_weight='balanced', random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        }
        self.trained_models = {}
        self.evaluation_results = []

    def get_stacking_model(self):
        """Tạo mô hình Stacking Classifier kết hợp các mô hình cơ sở."""
        estimators = [
            ('nb', MultinomialNB()),
            ('lr', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
            ('svm', SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42))
        ]
        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5
        )
        return stack_model

    def train_and_evaluate_all(self, X_train, y_train, X_test, y_test, target_names: List[str] = None):
        """Huấn luyện và đánh giá toàn bộ danh sách mô hình."""
        self.evaluation_results = []
        
        for name, model in self.models.items():
            print(f"-> Đang huấn luyện mô hình: {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            
            # Dự đoán trên tập test
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            self.evaluation_results.append({
                "Model": name,
                "Accuracy": acc,
                "F1 Macro": f1_macro,
                "F1 Weighted": f1_weighted
            })
            
        results_df = pd.DataFrame(self.evaluation_results).sort_values(by="F1 Macro", ascending=False)
        return results_df

    def plot_confusion_matrix(self, model_name: str, y_test, y_pred, labels: List[str], save_path: str = None):
        """Vẽ biểu đồ Confusion Matrix cho mô hình."""
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Dự đoán (Predicted)')
        plt.ylabel('Thực tế (Actual)')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Đã lưu biểu đồ tại: {save_path}")
        plt.show()

    def save_model(self, model_name: str, filepath: str):
        """Lưu mô hình đã huấn luyện."""
        if model_name in self.trained_models:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            joblib.dump(self.trained_models[model_name], filepath)
            print(f"Đã lưu mô hình '{model_name}' tại {filepath}")
        else:
            raise ValueError(f"Mô hình '{model_name}' chưa được huấn luyện.")
