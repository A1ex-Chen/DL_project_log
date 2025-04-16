def load_transcript(self, transcript_path):
    with open(transcript_path, 'r', encoding='utf-8') as transcript_file:
        transcript = transcript_file.read().replace('\n', '')
    return transcript
