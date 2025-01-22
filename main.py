from src.transcriber import AudioTranscriber

def main():
    input_directory = "./data/audio_files"
    output_file = "./data/output/transcription_result.txt"
    use_gpu = True

    transcriber = AudioTranscriber(use_gpu=use_gpu)

    try:
        transcriber.transcribe_directory(
            input_directory=input_directory,
            output_file=output_file,
            # file_pattern="講義"  # 特定のパターンのファイルのみを処理する場合
        )
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()