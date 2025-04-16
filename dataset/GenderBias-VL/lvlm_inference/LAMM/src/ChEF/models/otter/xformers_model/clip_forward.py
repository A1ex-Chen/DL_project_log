@add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling,
    config_class=CLIPVisionConfig)
def forward(self, pixel_values: Optional[torch.FloatTensor]=None,
    output_attentions: Optional[bool]=None, output_hidden_states: Optional[
    bool]=None, return_dict: Optional[bool]=None) ->Union[Tuple,
    BaseModelOutputWithPooling]:
    """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    return self.vision_model(pixel_values=pixel_values, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
