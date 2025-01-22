import pytest
import os
from unittest.mock import patch, MagicMock
from src.transcriber import AudioTranscriber

class TestAudioTranscriber:
    @pytest.fixture
    def transcriber(self):
        # TextFormatterクラスのモックを作成
        mock_formatter = MagicMock()
        mock_formatter.format_text = MagicMock(return_value="整形されたテストテキスト")

        with patch('whisper.load_model') as mock_load, \
             patch('src.transcriber.TextFormatter', return_value=mock_formatter):
            mock_model = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_load.return_value = mock_model
            return AudioTranscriber(use_gpu=False)

    def test_supported_formats(self, transcriber):
        # サポートされているフォーマットのテスト
        assert transcriber.is_supported_format("test.mp3")
        assert transcriber.is_supported_format("test.wav")
        assert transcriber.is_supported_format("test.FLAC")
        assert not transcriber.is_supported_format("test.txt")

    def test_transcribe_file(self, transcriber, tmp_path):
        # 単一ファイルの文字起こしテスト
        test_file = tmp_path / "test.mp3"
        test_file.write_text("")

        transcriber.model.transcribe.return_value = {"text": "テストテキスト"}

        result = transcriber.transcribe_file(str(test_file))
        assert result == "整形されたテストテキスト"

    def test_transcribe_directory(self, transcriber, tmp_path):
        # ディレクトリ処理のテスト
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # テストファイルの作成
        test_files = ["test1.mp3", "test2.wav", "test3.txt"]
        for file in test_files:
            (audio_dir / file).write_text("")

        output_file = tmp_path / "output.txt"

        transcriber.model.transcribe.return_value = {"text": "テストテキスト"}

        transcriber.transcribe_directory(
            input_directory=str(audio_dir),
            output_file=str(output_file)
        )

        assert output_file.exists()
        content = output_file.read_text()
        assert "test1.mp3" in content
        assert "test2.wav" in content
        assert "test3.txt" not in content

    def test_empty_directory(self, transcriber, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_file = tmp_path / "output.txt"

        transcriber.transcribe_directory(
            input_directory=str(empty_dir),
            output_file=str(output_file)
        )

        assert not output_file.exists()

    def test_file_pattern_filter(self, transcriber, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        test_files = ["lecture1.mp3", "lecture2.mp3", "other.mp3"]
        for file in test_files:
            (audio_dir / file).write_text("")

        output_file = tmp_path / "output.txt"

        transcriber.model.transcribe.return_value = {"text": "テストテキスト"}

        transcriber.transcribe_directory(
            input_directory=str(audio_dir),
            output_file=str(output_file),
            file_pattern="lecture"
        )

        content = output_file.read_text()
        assert "lecture1.mp3" in content
        assert "lecture2.mp3" in content
        assert "other.mp3" not in content
