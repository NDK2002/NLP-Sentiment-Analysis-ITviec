import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_reviews_data(filepath: str = None) -> pd.DataFrame:
    """Đọc dữ liệu reviews từ file excel."""
    if filepath is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(os.path.dirname(current_dir), 'data', 'raw', 'Reviews.xlsx')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại {filepath}")
    
    df = pd.read_excel(filepath)
    return df

def generate_wordcloud(text: str, title: str = "WordCloud", save_path: str = None):
    """Tạo và hiển thị WordCloud cho tập văn bản."""
    if not text.strip():
        print("Cảnh báo: Dữ liệu văn bản rỗng, không thể tạo WordCloud.")
        return

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=150,
        colormap='viridis'
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Đã lưu WordCloud tại: {save_path}")
    plt.show()
