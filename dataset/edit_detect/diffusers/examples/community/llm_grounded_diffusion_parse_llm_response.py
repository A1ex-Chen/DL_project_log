@classmethod
def parse_llm_response(cls, response, canvas_height=512, canvas_width=512):
    gen_boxes, bg_prompt, neg_prompt = cls._parse_response_with_negative(text
        =response)
    gen_boxes = sorted(gen_boxes, key=lambda gen_box: gen_box[0])
    phrases = [name for name, _ in gen_boxes]
    boxes = [cls.convert_box(box, height=canvas_height, width=canvas_width) for
        _, box in gen_boxes]
    return phrases, boxes, bg_prompt, neg_prompt
