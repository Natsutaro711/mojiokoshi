from transformers import T5Tokenizer, AutoModelForCausalLM
import torch
import re

class TextFormatter:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        if use_gpu and torch.cuda.is_available():
            print(f"GPUメモリ使用量(フォーマッター初期化前): {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"フォーマッター使用デバイス: {self.device}")

        # GPT-2モデルの初期化
        self.tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
        self.model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
        if self.device == "cuda":
            self.model = self.model.cuda()

    def add_punctuation(self, text):
        """基本的な句読点の追加"""
        # 文末表現のパターン
        sentence_end_patterns = [
            'です', 'ます', 'した', 'ません',
            'でしょう', 'ましょう', 'だ', 'である',
            'ください', 'なさい', '思います'
        ]

        # 読点を入れる基準となる接続詞や助詞
        comma_patterns = [
            'しかし', 'ただし', 'また', 'そして',
            'けれども', 'ところが', 'したがって',
            'ながら', 'けれど', 'のに', 'から'
        ]

        words = text.split()
        current_sentence = []
        sentences = []
        char_count = 0

        for word in words:
            current_sentence.append(word)
            char_count += len(word)

            # 読点の追加
            if any(pattern in word for pattern in comma_patterns):
                if not word.endswith('、'):
                    word = word + '、'
                current_sentence[-1] = word

            # 文末判定
            if any(word.endswith(end) for end in sentence_end_patterns):
                if not word.endswith('。'):
                    current_sentence[-1] = word + '。'
                sentences.append(''.join(current_sentence))
                current_sentence = []
                char_count = 0
            # 一定文字数で区切る（ただし、明確な文末表現がない場合）
            elif char_count >= 50:
                # 最後の単語が特定の助詞で終わる場合は読点を追加
                if any(word.endswith(p) for p in ['は', 'が', 'を', 'に', 'へ', 'で']):
                    current_sentence[-1] = word + '、'
                sentences.append(''.join(current_sentence))
                current_sentence = []
                char_count = 0

        # 残りの文字列を処理
        if current_sentence:
            text = ''.join(current_sentence)
            if not text.endswith('。'):
                text += '。'
            sentences.append(text)

        return '\n'.join(sentences)

    def format_text(self, input_text):
        """テキストの整形処理（メイン処理）"""
        # 基本的なクリーニング
        text = input_text.strip()
        text = re.sub(r'\s+', ' ', text)

        # 句読点の追加と文分割
        formatted_text = self.add_punctuation(text)

        # 段落分けの調整
        formatted_text = re.sub(r'\n{3,}', '\n\n', formatted_text)  # 過剰な改行を削除

        return formatted_text