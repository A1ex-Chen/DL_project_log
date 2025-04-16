def convert_longformer_qa_checkpoint_to_pytorch(longformer_model: str,
    longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str
    ):
    longformer = LongformerModel.from_pretrained(longformer_model)
    lightning_model = LightningModel(longformer)
    ckpt = torch.load(longformer_question_answering_ckpt_path, map_location
        =torch.device('cpu'))
    lightning_model.load_state_dict(ckpt['state_dict'])
    longformer_for_qa = LongformerForQuestionAnswering.from_pretrained(
        longformer_model)
    longformer_for_qa.longformer.load_state_dict(lightning_model.model.
        state_dict())
    longformer_for_qa.qa_outputs.load_state_dict(lightning_model.qa_outputs
        .state_dict())
    longformer_for_qa.eval()
    longformer_for_qa.save_pretrained(pytorch_dump_folder_path)
    print('Conversion successful. Model saved under {}'.format(
        pytorch_dump_folder_path))
