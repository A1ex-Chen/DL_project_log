def gather_transcripts(transcript_list, transcript_len_list, detokenize):
    return [detokenize(t[:l].long().cpu().numpy().tolist()) for txt, lens in
        zip(transcript_list, transcript_len_list) for t, l in zip(txt, lens)]
