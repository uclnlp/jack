
def raw_data_batcher()
    def batch_generator():
    todo = list(range(len(q_ids)))
    self._rng.shuffle(todo)
    while todo:
        support_lengths = list()
        question_lengths = list()
        wiq = list()
        spans = list()
        span2question = []
        offsets = []

        unique_words, unique_word_lengths, question2unique, support2unique = \
            self.unique_words(q_tokenized, s_tokenized, todo[:self.batch_size])

        # we have to create batches here and cannot precompute them because of the batch-specific wiq feature
        for i, j in enumerate(todo[:self.batch_size]):
            support = s_ids[j]
            for k in range(len(support)):
                emb_supports[i, k] = super().get_emb(support[k])
            question = q_ids[j]
            for k in range(len(question)):
                emb_questions[i, k] = super().get_emb(question[k])
            support_lengths.append(s_lengths[j])
            question_lengths.append(q_lengths[j])
            spans.extend(answer_spans[j])
            span2question.extend(i for _ in answer_spans[j])
            wiq.append(word_in_question[j])
            offsets.append(token_offsets[j])

            yield unique_words, unique_word_lengths, question2unique,
            support2unique,
            emb_supports[:batch_size, :max(support_lengths), :],
            support_lengths,
            emb_questions[:batch_size, :max(question_lengths), :],
            question_lengths,
            wiq, spans, [] if is_eval else [s[0] for s in spans],
            span2question, span2question, 1.0 if is_eval else 1.0 - self.dropout,
            is_eval, offsets
    return GeneratorWithRestart(batch_generator)
