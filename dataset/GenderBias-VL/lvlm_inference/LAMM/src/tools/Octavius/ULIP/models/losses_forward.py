def forward(self, object_name, outputs):
    pc_embed = outputs['pc_embed']
    text_embed = outputs['text_embed']
    image_embed = outputs['image_embed']
    logit_scale = outputs['logit_scale']
    local_batch_size = pc_embed.size(0)
    if local_batch_size != self.last_local_batch_size:
        self.labels = local_batch_size * utils.get_rank() + torch.arange(
            local_batch_size, device=pc_embed.device)
        self.last_local_batch_size = local_batch_size
    if self.args.use_memory_bank:
        obj_image_memory_bank = [None] * local_batch_size
        for i in range(local_batch_size):
            negative_obj_image = self.obj_class_memory_bank[self.
                obj_classes[object_name[i]]]
            negative_obj_image = np.array(negative_obj_image)[np.random.
                choice(range(len(negative_obj_image)), self.args.
                memory_bank_size - 1, replace=False)]
            negative_obj_image = torch.from_numpy(negative_obj_image).cuda()
            obj_image_memory_bank[i] = torch.cat((image_embed[i:i + 1],
                negative_obj_image), dim=0).unsqueeze(0)
        obj_image_memory_bank = torch.cat(obj_image_memory_bank, dim=0).cuda()
        text_memory_bank = [None] * local_batch_size
        for i in range(local_batch_size):
            negative_obj_text = self.text_memory_bank[self.obj_classes[
                object_name[i]]]
            negative_obj_text = np.array(negative_obj_text)[np.random.
                choice(range(len(negative_obj_text)), self.args.
                memory_bank_size - 1, replace=False)]
            negative_obj_text = torch.from_numpy(negative_obj_text).cuda()
            text_memory_bank[i] = torch.cat((text_embed[i:i + 1],
                negative_obj_text), dim=0).unsqueeze(0)
        text_memory_bank = torch.cat(text_memory_bank, dim=0).cuda()
    pc_embed = F.normalize(pc_embed, dim=-1, p=2)
    text_embed = F.normalize(text_embed, dim=-1, p=2)
    image_embed = F.normalize(image_embed, dim=-1, p=2)
    if self.args.use_memory_bank:
        obj_image_memory_bank = F.normalize(obj_image_memory_bank, dim=-1, p=2)
    if self.args.use_memory_bank:
        text_memory_bank = F.normalize(text_memory_bank, dim=-1, p=2)
    pc_embed_all, text_embed_all, image_embed_all = utils.all_gather_batch([
        pc_embed, text_embed, image_embed])
    if self.args.use_memory_bank:
        pc_embed = pc_embed.unsqueeze(1)
        obj_image_memory_bank = obj_image_memory_bank.transpose(2, 1)
        logits_per_pc_image = logit_scale * (pc_embed @ obj_image_memory_bank
            ).squeeze(1)
        text_memory_bank = text_memory_bank.transpose(2, 1)
        logits_per_pc_text = logit_scale * (pc_embed @ text_memory_bank
            ).squeeze(1)
        self.memory_bank_labels = torch.zeros_like(self.labels)
        text_loss = F.cross_entropy(logits_per_pc_text, self.memory_bank_labels
            )
        image_loss = F.cross_entropy(logits_per_pc_image, self.
            memory_bank_labels)
        loss = text_loss + image_loss
    else:
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()
        text_loss = (F.cross_entropy(logits_per_pc_text, self.labels) + F.
            cross_entropy(logits_per_text_pc, self.labels)) / 2
        image_loss = (F.cross_entropy(logits_per_pc_image, self.labels) + F
            .cross_entropy(logits_per_image_pc, self.labels)) / 2
        loss = text_loss + image_loss
    with torch.no_grad():
        pred = torch.argmax(logits_per_pc_text, dim=-1)
        if self.args.use_memory_bank:
            correct = pred.eq(self.memory_bank_labels).sum()
        else:
            correct = pred.eq(self.labels).sum()
        pc_text_acc = 100 * correct / local_batch_size
        pred = torch.argmax(logits_per_pc_image, dim=-1)
        if self.args.use_memory_bank:
            correct = pred.eq(self.memory_bank_labels).sum()
        else:
            correct = pred.eq(self.labels).sum()
        pc_image_acc = 100 * correct / local_batch_size
    return {'loss': loss, 'text_loss': text_loss, 'image_loss': image_loss,
        'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}
