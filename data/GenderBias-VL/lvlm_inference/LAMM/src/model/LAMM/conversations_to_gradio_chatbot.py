def to_gradio_chatbot(self):
    ret = []
    for i, (role, msg) in enumerate(self.messages[self.offset:]):
        if i % 2 == 0:
            if type(msg) is tuple:
                import base64
                from io import BytesIO
                msg, image, image_process_mode = msg
                max_hw, min_hw = max(image.size), min(image.size)
                aspect_ratio = max_hw / min_hw
                max_len, min_len = 800, 400
                shortest_edge = int(min(max_len / aspect_ratio, min_len,
                    min_hw))
                longest_edge = int(shortest_edge * aspect_ratio)
                W, H = image.size
                if H > W:
                    H, W = longest_edge, shortest_edge
                else:
                    H, W = shortest_edge, longest_edge
                image = image.resize((W, H))
                buffered = BytesIO()
                image.save(buffered, format='JPEG')
                img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                img_str = (
                    f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    )
                msg = msg.replace('<image>', img_str)
            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret
