# -*- coding: utf-8 -*-

import tensorflow as tf

from jtr.projects.modelF.structs import Identifier
from collections import namedtuple


def tok(string):
    return string.split(" ")


WHITESPACE = '[WS]'

Transformation = namedtuple('Transformation', ['transformations', 'transformation_probs',
                                               'total_score', 'summarized', 'summarized_decoded'])
ProofStep = namedtuple('ProofStep', ['extractions', 'questions', 'terminate_prob', 'current_question'])


class Model1:
    def __init__(self, max_num_tokens, max_steps=1):
        self.vocab = Identifier()
        self.embeddings = tf.diag(tf.ones(max_num_tokens))
        self.whitespace_repr = tf.gather(self.embeddings, self.vocab[WHITESPACE])  # [repr_dim]
        self.cost_for_remainder = 1 - tf.gather(self.embeddings, self.vocab['_']) - self.whitespace_repr

        self.wh_token = tf.gather(self.embeddings, self.vocab['_'])  # [repr_dim]
        self.translate_token = tf.gather(self.embeddings, self.vocab['isa'])  # [repr_dim]
        self.softmax_slope = 2.0
        self.statement_placeholder = tf.placeholder(tf.int32, (None, None))
        self.question_placeholder = tf.placeholder(tf.int32, (None, None))
        self.max_steps = max_steps

        embedded_statements = self.embed_statements(self.statement_placeholder)
        embedded_questions = self.embed_statements(self.question_placeholder)
        self.match_result = self.simple_extract_or_translate(embedded_questions, embedded_statements)
        self.match_all_result = self.match_all(embedded_statements, embedded_questions, decode_kb=False)
        self.match_all_result_decoded = self.match_all(embedded_statements, embedded_questions, decode_kb=True)
        self.match_iteratively_result = self.inference_steps(embedded_statements, embedded_questions, max_steps)
        self.match_iteratively_result_decoded = self.inference_steps(embedded_statements, embedded_questions, max_steps,
                                                                     decode_kb=True)
        # self.match_iteratively_decoded = [(self.decode(step[0]), self.decode(step[3])) for step in
        #                                   self.match_iteratively_result]

        self.sess = tf.Session()

    def repr_text_batch(self, texts):
        tokenized = [tok(text) for text in texts]
        max_length = max([len(tokens) for tokens in tokenized])
        result = [[self.vocab[tokens[i]] if i < len(tokens) else self.vocab[WHITESPACE] for i in range(0, max_length)]
                  for
                  tokens in tokenized]
        return result

    def query(self, questions, statements):
        ans, ans_score, q, q_score = self.match_result
        ans_decoded, ans_score_result, q_decoded, q_score_result = \
            self.sess.run((self.decode(ans), ans_score, self.decode(q), q_score),
                          feed_dict=self.to_feed_dict(statements, questions))
        return self.to_strings(ans_decoded), ans_score_result, self.to_strings(q_decoded), q_score_result

    def query_all_decoded(self, questions, statements):
        ext, que = self.match_all_result_decoded
        result = self.sess.run(ext + que, feed_dict=self.to_feed_dict(statements, questions))
        return Transformation(self.to_kb_strings(result[0]), result[1], result[2], result[3],
                              self.to_strings(result[4])), \
               Transformation(self.to_kb_strings(result[5]), result[6], result[7], result[8],
                              self.to_strings(result[9]))

    def query_iteratively_decoded(self, questions, statements):
        steps = self.match_iteratively_result_decoded
        flattened_steps = [op for step in steps for op in step.extractions + step.questions +
                           (step.terminate_prob, step.current_question)]
        feed_dict = self.to_feed_dict(statements, questions)
        step_results = self.sess.run(flattened_steps, feed_dict=feed_dict)
        step_length = 12
        results = []
        for i in range(0, len(steps)):
            result = step_results[i * step_length: (i + 1) * step_length]
            trans_ans = Transformation(self.to_kb_strings(result[0]), result[1], result[2], result[3],
                                       self.to_strings(result[4]))
            trans_que = Transformation(self.to_kb_strings(result[5]), result[6], result[7], result[8],
                                       self.to_strings(result[9]))
            term_prob = result[10]
            current_question = result[11]
            results.append(ProofStep(trans_ans, trans_que, term_prob, current_question))

        return results

    def simple_extract_or_translate(self, questions, statements):
        # questions: [batch_size, length, repr_dim]
        # statements: [batch_size, length, repr_dim]
        # return: extraction result,extraction score, translation result, translation score
        # calculate total token-by-token match score
        match_score = tf.reduce_sum(questions * statements, [1, 2])  # [batch_size]

        # find WH token
        wh_scores = tf.reduce_sum(questions * self.wh_token, 2)  # [batch_size, length]
        wh_probs = tf.nn.softmax(wh_scores)  # [batch_size, length]

        # extract answer token
        answer_token = tf.reduce_sum(statements * tf.expand_dims(wh_probs, 2), 1)  # [batch_size, repr_dim]

        # answer should be a sequence, append zeros
        padding = tf.zeros(tf.shape(statements) - [0, 1, 0])
        padding_ws = tf.tile(tf.expand_dims(tf.expand_dims(self.whitespace_repr, 0), 0),
                             tf.shape(statements) * [1, 1, 0] + [0, -1, 1])
        answer = tf.concat([tf.expand_dims(answer_token, 1), padding_ws], 1)

        # check for TR token at index 1
        tr_match_score = tf.reduce_sum(statements[:, 1, :] * self.translate_token, 1)  # [batch_size]
        # A => Y
        lhs = statements[:, 0:1, :]  # [batch_size, 1, repr_dim]
        rhs = statements[:, 2:3, :]

        # find best match with rhs in question
        lhs_match_scores = tf.reduce_sum(questions * lhs, 2)  # [batch_size, length]
        lhs_prob = tf.nn.softmax(lhs_match_scores * 5)  # [batch_size, length]
        lhs_prob_expanded = tf.expand_dims(lhs_prob, 2)  # [batch_size, length, 1]

        # then replace with lhs
        replacement = rhs * lhs_prob_expanded
        to_remove = lhs * lhs_prob_expanded
        new_questions = questions - to_remove + replacement

        # translation score
        tr_score = tf.reduce_sum(lhs_match_scores, 1) + tr_match_score

        return answer, match_score, new_questions, tr_score

    def embed_statements(self, statements):
        return tf.gather(self.embeddings, statements)

    def to_feed_dict(self, statements, questions):
        text_repr = self.repr_text_batch(statements + questions)
        statement_repr, question_repr = text_repr[:len(statements)], text_repr[len(statements):]
        return {self.statement_placeholder: statement_repr, self.question_placeholder: question_repr}

    def match_all(self, kb, questions, decode_kb, decode_answer=True):
        def repr_kb(input):
            return self.decode_kb(input) if decode_kb else input

        def repr_answer(input):
            return self.decode(input) if decode_answer else input

        # kb: [kb_size, max_length, repr_dim]
        # questions [batch_size, max_length, repr_dim]
        # turn kb into [batch_size, kb_size, max_length, repr_dim]
        expanded_kb = tf.expand_dims(kb, 0)
        expanded_questions = tf.expand_dims(questions, 1)
        tiled_kb = tf.tile(expanded_kb, tf.shape(expanded_questions) * [1, 0, 0, 0] + [0, 1, 1, 1])
        tiled_questions = tf.tile(expanded_questions, tf.shape(expanded_kb) * [0, 1, 0, 0] + [1, 0, 1, 1])

        # now flatten
        new_dim = tf.shape(kb)[0:1] * tf.shape(questions)[0:1]
        new_shape = tf.concat([new_dim, tf.shape(kb)[1:]], 0)
        batch_kb_shape = tf.shape(tiled_kb)[0:2]
        flat_kb = tf.reshape(tiled_kb, new_shape)
        flat_questions = tf.reshape(tiled_questions, new_shape)

        answers, match_scores, new_questions, tr_scores = self.simple_extract_or_translate(flat_questions, flat_kb)

        # extraction
        def aggregate_and_score(answers, match_scores, scale=1.0):
            answers_reshaped = tf.reshape(answers, tf.shape(tiled_kb))  # [batch_size, kb_size, max_length, repr_dim]
            match_scores_reshaped = scale * tf.reshape(match_scores, batch_kb_shape)  # [batch_size, kb_size]
            match_probs = tf.nn.softmax(match_scores_reshaped)  # [batch_size, kb_size]
            match_probs_expanded = tf.expand_dims(tf.expand_dims(match_probs, 2), 3)  # [batch_size, kb_size, 1, 1]
            weighted_answer = tf.reduce_sum(match_probs_expanded * answers_reshaped, 1)
            global_match_score = tf.reduce_sum(match_scores_reshaped, 1)
            # return weighted_answer, global_match_score, match_probs
            return Transformation(repr_kb(answers_reshaped), match_scores_reshaped, global_match_score,
                                  weighted_answer, repr_answer(weighted_answer))

        extractions = aggregate_and_score(answers, match_scores)
        questions = aggregate_and_score(new_questions, tr_scores, scale=2.)

        return extractions, questions

    def inference_steps(self, kb, questions, num_steps=1, decode_kb=False, decode_answer=True):
        current_questions = questions
        steps = []
        for step in range(0, num_steps):
            extractions, questions = self.match_all(kb, current_questions, decode_kb, decode_answer)
            # if global_match_score >> global_translation_score we should never change the question again
            prob_translate = tf.sigmoid(1.0 * (questions.total_score - extractions.total_score))
            expanded_prob = tf.expand_dims(tf.expand_dims(prob_translate, 1), 1)
            current_questions = expanded_prob * questions.summarized + (1.0 - expanded_prob) * current_questions
            #         current_questions = global_translation
            proof_step = ProofStep(extractions, questions, 1.0 - prob_translate, current_questions)
            steps.append(proof_step)

        return steps

    def decode(self, statements):
        # statements: [batch_size, length, repr_dim]
        # -> [batch_size, length, 1, repr_dim]
        compare_all = tf.reduce_sum(tf.expand_dims(statements, 2) * self.embeddings, 3)
        _, indices = tf.nn.top_k(compare_all)
        return indices

    def decode_kb(self, kb):
        # statements: [batch_size, kb_length, length, repr_dim]
        # -> [batch_size, length, kb_length, 1, repr_dim]
        compare_all = tf.reduce_sum(tf.expand_dims(kb, 3) * self.embeddings, 4)
        _, indices = tf.nn.top_k(compare_all)
        return indices

    def to_strings(self, indices):
        results = []
        for seq in indices:
            sentence = []
            for token in seq:
                word = self.vocab.key_by_id(token[0])
                if word != WHITESPACE:
                    sentence.append(self.vocab.key_by_id(token[0]))
            results.append(sentence)
        return results

    def to_kb_strings(self, indices):
        results = []
        for row in indices:
            row_result = []
            for seq in row:
                sentence = []
                for token in seq:
                    word = self.vocab.key_by_id(token[0])
                    if word != WHITESPACE:
                        sentence.append(self.vocab.key_by_id(token[0]))
                row_result.append(sentence)
            results.append(row_result)
        return results


if __name__ == "__main__":
    model = Model1(10, 3)
    # print(model.query(["eagle can _"], ["eagle can fly"]))
    # print(model.query(["eagle can _"], ["eagle isa bird"]))
    # print(model.query_all_decoded(["eagle can _", "fish can _"], ["eagle can fly", "fish can swim"]))
    # print(model.query_iteratively_decoded(["eagle can _"], ["eagle isa bird"]))
    # print(model.query_iteratively_decoded(["eagle can _"], ["eagle isa bird"]))
    print("----")
    # print(model.query_iteratively_decoded(["eagle can _"], ["eagle isa bird", "bird can fly"]))
    # for step in model.query_iteratively_decoded(["eagle can _"], ["eagle isa bird", "bird can fly"]):
    #     print(step)
    #     # print(model.query_iteratively(["eagle can _"], ["eagle can fly", "fish can swim"]))
    #     # print(model.query_iteratively(["eagle can _", "fish can _"], ["eagle can fly", "fish can swim"]))
    for step in model.query_iteratively_decoded(["eagle can _"], ["eagle isa bird", "bird isa animal", "animal can eat"]):
        print(step)
        # print(model.query_iteratively(["eagle can _"], ["eagle can fly", "fish can swim"]))
        # print(model.query_iteratively(["eagle can _", "fish can _"], ["eagle can fly", "fish can swim"]))
