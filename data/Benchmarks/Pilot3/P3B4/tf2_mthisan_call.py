def call(self, docs):
    batch_size = tf.shape(docs)[0]
    words_per_line = tf.math.count_nonzero(docs, 2, dtype=tf.int32)
    max_words = tf.reduce_max(words_per_line)
    lines_per_doc = tf.math.count_nonzero(words_per_line, 1, dtype=tf.int32)
    max_lines = tf.reduce_max(lines_per_doc)
    num_words = words_per_line[:, :max_lines]
    num_words = tf.reshape(num_words, (-1,))
    doc_input_reduced = docs[:, :max_lines, :max_words]
    skip_lines = tf.not_equal(num_words, 0)
    count_lines = tf.reduce_sum(tf.cast(skip_lines, tf.int32))
    mask_words = tf.sequence_mask(num_words, max_words)[skip_lines]
    mask_words = tf.tile(tf.expand_dims(mask_words, 1), [1, self.
        attention_heads, 1])
    mask_lines = tf.sequence_mask(lines_per_doc, max_lines)
    mask_lines = tf.tile(tf.expand_dims(mask_lines, 1), [1, self.
        attention_heads, 1])
    doc_input_reduced = tf.reshape(doc_input_reduced, (-1, max_words))[
        skip_lines]
    word_embeds = self.embedding(doc_input_reduced)
    word_embeds = self.word_drop(word_embeds, training=self.training)
    word_q = self._split_heads(self.word_Q(word_embeds), count_lines)
    word_k = self._split_heads(self.word_K(word_embeds), count_lines)
    word_v = self._split_heads(self.word_V(word_embeds), count_lines)
    word_self_out = self.word_self_att([word_q, word_v, word_k], [
        mask_words, mask_words], training=self.training)
    word_target = tf.tile(self.word_target, [count_lines, 1, 1, 1])
    word_targ_out = self.word_targ_att([word_target, word_self_out,
        word_self_out], [None, mask_words], training=self.training)
    word_targ_out = tf.transpose(word_targ_out, perm=[0, 2, 1, 3])
    line_embeds = tf.scatter_nd(tf.where(skip_lines), tf.reshape(
        word_targ_out, (count_lines, self.attention_size)), (batch_size *
        max_lines, self.attention_size))
    line_embeds = tf.reshape(line_embeds, (batch_size, max_lines, self.
        attention_size))
    line_embeds = self.line_drop(line_embeds, training=self.training)
    line_q = self._split_heads(self.line_Q(line_embeds), batch_size)
    line_k = self._split_heads(self.line_K(line_embeds), batch_size)
    line_v = self._split_heads(self.line_V(line_embeds), batch_size)
    line_self_out = self.line_self_att([line_q, line_v, line_k], [
        mask_lines, mask_lines], training=self.training)
    line_target = tf.tile(self.line_target, [batch_size, 1, 1, 1])
    line_targ_out = self.line_targ_att([line_target, line_self_out,
        line_self_out], [None, mask_lines], training=self.training)
    line_targ_out = tf.transpose(line_targ_out, perm=[0, 2, 1, 3])
    doc_embeds = tf.reshape(line_targ_out, (batch_size, self.attention_size))
    doc_embeds = self.doc_drop(doc_embeds, training=self.training)
    logits = []
    for lIndex in self.classify_layers:
        logits.append(lIndex(doc_embeds))
    return logits
