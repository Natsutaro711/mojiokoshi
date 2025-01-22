from typing import List, Optional
import whisper
import torch
import os
from src.text_formatter import TextFormatter

class AudioTranscriber:
    def __init__(self, model_size: str = "large", use_gpu: bool = False):
        self.use_gpu = use_gpu
        if use_gpu:
            if not torch.cuda.is_available():
                print("警告: GPUが利用できません。CPUを使用します。")
            else:
                print(f"利用可能なGPU: {torch.cuda.get_device_name()}")
                print(f"現在のGPUメモリ使用量: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")

        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"文字起こし使用デバイス: {self.device}")

        self.model = whisper.load_model(model_size).to(self.device)
        self.formatter = TextFormatter(use_gpu)

        self.SUPPORTED_FORMATS = (
            '.mp3',   # MP3
            '.aac',   # AAC
            '.ogg', '.oga',  # OGG
            '.wma',   # WMA
            '.opus',  # OPUS
            '.wav',   # WAV
            '.flac',  # FLAC
            '.aiff', '.aif'  # AIFF
        )

        # GPUメモリの効率化設定
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()  # キャッシュをクリア
            torch.backends.cudnn.benchmark = True  # CUDNN最適化

    def is_supported_format(self, filename: str) -> bool:
        return filename.lower().endswith(self.SUPPORTED_FORMATS)

    def transcribe_file(self, file_path: str) -> str:
        """単一ファイルの文字起こしを行う"""
        print(f"文字起こし中: {os.path.basename(file_path)}")
        result = self.model.transcribe(file_path, language="Japanese")

        print(f"テキスト整形中: {os.path.basename(file_path)}")
        return self.formatter.format_text(result["text"])

    def transcribe_directory(self,
                           input_directory: str,
                           output_file: str,
                           file_pattern: Optional[str] = None) -> None:
        """ディレクトリ内の音声ファイルを文字起こし"""
        if not os.path.exists(input_directory):
            raise FileNotFoundError(f"Directory not found: {input_directory}")

        combined_text = ""
        processed_files = []

        for filename in sorted(os.listdir(input_directory)):
            if not self.is_supported_format(filename):
                continue

            if file_pattern and file_pattern not in filename:
                continue

            file_path = os.path.join(input_directory, filename)
            transcribed_text = self.transcribe_file(file_path)
            combined_text += f"\n\n--- {filename} ---\n\n{transcribed_text}"
            processed_files.append(filename)

        if not processed_files:
            print("処理対象のファイルが見つかりませんでした。")
            return

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"\n文字起こし完了！結果は {output_file} に保存されました。")
        print(f"処理したファイル: {', '.join(processed_files)}")

    def __str__(self) -> str:
        return f"AudioTranscriber(device={self.device}, supported_formats={self.SUPPORTED_FORMATS})"