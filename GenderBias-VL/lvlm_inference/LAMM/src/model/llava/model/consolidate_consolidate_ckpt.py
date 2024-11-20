def consolidate_ckpt(src_path, dst_path):
    print('Loading model')
    auto_upgrade(src_path)
    src_model = AutoModelForCausalLM.from_pretrained(src_path, torch_dtype=
        torch.float16, low_cpu_mem_usage=True)
    src_tokenizer = AutoTokenizer.from_pretrained(src_path, use_fast=False)
    src_model.save_pretrained(dst_path)
    src_tokenizer.save_pretrained(dst_path)
