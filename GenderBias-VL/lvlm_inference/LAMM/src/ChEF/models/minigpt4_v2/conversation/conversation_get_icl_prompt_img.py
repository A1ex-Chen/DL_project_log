def get_icl_prompt_img(self, question, conv, img_list, ice, ice_imgs_emb,
    index, incontext_cfg):
    if incontext_cfg['add_sysmsg']:
        conv.system += incontext_cfg['sysmsg']
    if incontext_cfg['use_pic']:
        img_list = ice_imgs_emb[index] + img_list
        if incontext_cfg['mult_conversations']:
            for i in range(len(ice_imgs_emb[index])):
                if not isinstance(ice[i]['gt_answers'], list):
                    conv.messages[-1][-1] += ice[i]['question']
                    conv.append_message(conv.roles[1], ice[i]['gt_answers'])
                else:
                    conv.messages[-1][-1] += ice[i]['question']
                    conv.append_message(conv.roles[1], ice[i]['gt_answers'][0])
                conv.append_message(conv.roles[0], '<Img><ImageHere></Img>')
            conv.messages[-1][-1] += question
        else:
            for i in range(len(ice_imgs_emb[index])):
                if not isinstance(ice[i]['gt_answers'], list):
                    conv.messages[-1][-1
                        ] += f"{ice[i]['question']}: {ice[i]['gt_answers']}."
                else:
                    conv.messages[-1][-1
                        ] += f"{ice[i]['question']}: {ice[i]['gt_answers'][0]}."
                conv.messages[-1][-1] += '<Img><ImageHere></Img>'
            conv.messages[-1][-1] += question
    else:
        icl_question = ''
        for j in range(incontext_cfg['ice_num']):
            if not isinstance(ice[j]['gt_answers'], list):
                icl_question += (
                    f"{ice[j]['question']}: {ice[j]['gt_answers']}. ")
            else:
                icl_question += (
                    f"{ice[j]['question']}: {ice[j]['gt_answers'][0]}. ")
        icl_question += f'{question}: '
        self.ask(icl_question, conv)
    return img_list
