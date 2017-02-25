

import _gdynet as dy
#dy.init()
#import _dynet as dy
dyparams = dy.DynetParams()
dyparams.set_mem(10000)
dyparams.init()
import random
import codecs
import numpy as np


### FILE PATHS
train_en_path = '/home/ubuntu/shrimai/11731/data/en-de/train.en-de.low.filt.en'
train_de_path = '/home/ubuntu/shrimai/11731/data/en-de/train.en-de.low.filt.de'

valid_en_path = '/home/ubuntu/shrimai/11731/data/en-de/valid.en-de.low.en'
valid_de_path = '/home/ubuntu/shrimai/11731/data/en-de/valid.en-de.low.de'

test_en_path = '/home/ubuntu/shrimai/11731/data/en-de/test.en-de.low.en'
test_de_path = '/home/ubuntu/shrimai/11731/data/en-de/test.en-de.low.de'

TRAIN_SWITCH = False
load_model_path = '/home/ubuntu/shrimai/11731/batched_models/novalid_e5h5a2model_20_3067'

### To read the Files ###
class Corpus(object):
    def __init__(self, filename):
        with codecs.open(filename, 'r') as inp:
            data = inp.readlines()

        self.data = [x.strip().lower() for x in data]
        self.max_len = 0

    def create_dict(self):
        words = []
        for i, sent in enumerate(self.data):
            split_sent = sent.split()
            words.extend(split_sent)

        words = list(set(words))
        words.append('<UNK>')
        words = sorted(words)
        words.insert(0, '<s>')
        words.insert(0, '<EOS>')

        idx2word = {v: k for v, k in enumerate(words)}
        word2idx = {k: v for v, k in enumerate(words)}
        return word2idx, idx2word

    def tokenize(self):
        word2idx, idx2word = self.create_dict()

        final = []
        for i, sent in enumerate(self.data):
            words = sent.split()
            words.insert(0, '<s>')
            words.append('<EOS>')
            index = [word2idx[x] for x in words]
            final.append(index)
        return word2idx, idx2word, final

def prepare_data():
    train_en = Corpus(train_en_path)
    word2idx_en, id2xword_en, data_en = train_en.tokenize()
    #print word2idx_en['<UNK>']

    train_de = Corpus(train_de_path)
    word2idx_de, idx2word_de, data_de = train_de.tokenize()

    return word2idx_en, id2xword_en, data_en, idx2word_de, word2idx_de, data_de

### Here are all the vocabs and dictionaries used in the code ###
word2idx_en, idx2word_en, data_en, idx2word_de, word2idx_de, data_de = prepare_data()

### Hyper Paramters for Model ##
vocab_size_en = len(word2idx_en.keys())
vocab_size_de = len(word2idx_de.keys())

no_layers = 2
embedding_size = 500
hidden_size = 500
attention_size = 200
batch_size = 32
no_epochs = 15
valid_batch_size = 5
### Make False if you don't want to use dropout ###
dropout_config = True
dropout_val = 0.2

model = dy.Model()

### Preparing the validation and test data ###
def get_data(filename):
        with codecs.open(filename, 'r') as inp:
            lines = [x.strip().split() for x in inp.readlines()]
        return lines

def get_idx(sent_list, dic):
        idx = []
        total_toks = 0
        for sent in sent_list:
            wids = []
            total_toks = total_toks + len(sent)
            wids.insert(0, dic['<s>'])
            for word in sent:
                if dic.has_key(word):
                    wids.append(dic[word])
                else:
                    wids.append(dic['<UNK>'])
            wids.append(dic['<EOS>'])
            idx.append(wids)

        return idx, total_toks


def prepare_test_data(enc_file, dec_file):
        encoder_test_file, decoder_test_file = enc_file, dec_file
        encoder_dict, decoder_dict = word2idx_de, word2idx_en

        encoder_sentences = get_data(encoder_test_file)
        decoder_sentences = get_data(decoder_test_file)

        encoder_wids, _ = get_idx(encoder_sentences, encoder_dict)
        decoder_wids, total_dec_toks = get_idx(decoder_sentences, decoder_dict)

        return encoder_sentences, encoder_wids, decoder_sentences, decoder_wids, total_dec_toks

_, valid_enc_ids, _, valid_dec, total_vtoks = prepare_test_data(valid_de_path, valid_en_path)


if TRAIN_SWITCH:
    enc_fwd_lstm = dy.LSTMBuilder(no_layers, embedding_size, hidden_size, model)
    enc_bwd_lstm = dy.LSTMBuilder(no_layers, embedding_size, hidden_size, model)
    dec_lstm = dy.LSTMBuilder(no_layers, hidden_size*2+embedding_size, hidden_size, model)

    input_lookup = model.add_lookup_parameters((vocab_size_de, embedding_size))
    attention_w1 = model.add_parameters( (attention_size, hidden_size*2))
    attention_w2 = model.add_parameters( (attention_size, hidden_size*no_layers*2))
    attention_v = model.add_parameters( (1, attention_size))
    decoder_w = model.add_parameters( (vocab_size_en, 3*hidden_size))
    decoder_b = model.add_parameters( (vocab_size_en))
    output_lookup = model.add_lookup_parameters((vocab_size_en, embedding_size))
else:
    [enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup,
 attention_w1, attention_w2, attention_v, decoder_w, decoder_b] = model.load(load_model_path)
    _, encoder_test_wids, _, _, _ = prepare_test_data(test_de_path, test_en_path)
    #_, encoder_test_wids, _, _, _ = prepare_test_data(valid_de_path, valid_en_path)

### Main Class ###
class AttentionModel():

    def __init__(self):
        pass

    def pad_zero(self, batchify):
        S = word2idx_en['<EOS>']

        max_len = max(map(lambda x: len(x), batchify))

        wids = []
        masks = []
        for i in range(max_len):
            wids.append([(sent[i] if len(sent)>i else S) for sent in batchify])
            mask = [(1 if len(sent) > i else 0) for sent in batchify]
            masks.append(mask)

        return wids, masks

    def get_batch(self, train_set, batch_size):

        total = len(train_set)
        b = total/batch_size
        mini_batches = []
        for i in range(0, total, batch_size):
            mini_batches.append(train_set[i:i+batch_size])
        mini_batches.append(train_set[i:])

        return mini_batches

    def get_decoded(self, input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        encoded = self.encode_sentence(enc_fwd_lstm, enc_bwd_lstm, input_sentence)
        return self.myDecode(dec_lstm, encoded, output_sentence)

    def embed_sentence(self, sentence, generation=False):
        if generation:
            embeddings = [input_lookup[word] for word in sentence]
        else:
            embeddings = [dy.lookup_batch(input_lookup, wids) for wids in sentence]
        return embeddings

    def encode_sentence(self, enc_fwd_lstm, enc_bwd_lstm, sentence):
        dy.renew_cg()

        enc_sents, _ = self.pad_zero(sentence)
        sent_embed = self.embed_sentence(enc_sents)
        sentrev_embed = sent_embed[::-1]

        enc_fwd = enc_fwd_lstm.initial_state()
        enc_bwd = enc_bwd_lstm.initial_state()
        fwd_vectors = enc_fwd.transduce(sent_embed)
        bwd_vectors = enc_bwd.transduce(sentrev_embed)
        bwd_vectors = bwd_vectors[::-1]

        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def encode_generation(self, enc_fwd_lstm, enc_bwd_lstm, sentence):
        sentence_rev = list(reversed(sentence))

        enc_fwd = enc_fwd_lstm.initial_state()
        enc_bwd = enc_bwd_lstm.initial_state()
        fwd_vectors = self.run_lstm(enc_fwd, sentence)
        bwd_vectors = self.run_lstm(enc_bwd, sentence_rev)
        bwd_vectors = list(reversed(bwd_vectors))

        vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def run_lstm(self, init_state, input_vecs):
        s = init_state
        out_vectors = []

        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def myDecode(self, dec_lstm, h_encodings, target_sents):
        w = dy.parameter(decoder_w)
        b = dy.parameter(decoder_b)
        w1 = dy.parameter(attention_w1)

        dec_wrds, dec_mask = self.pad_zero(target_sents)
        curr_bsize = len(target_sents)
        h_len = len(h_encodings)

        H_source = dy.concatenate_cols(h_encodings)
        s = dec_lstm.initial_state()
        ctx_t0 = dy.vecInput(hidden_size * 2)
        w1dt = w1 * H_source

        loss = []

        #print curr_bsize
        for sent in range(1, len(dec_wrds)):
            last_output_embeddings = dy.lookup_batch(output_lookup, dec_wrds[sent-1])
            x = dy.concatenate([ctx_t0, last_output_embeddings])
            s = s.add_input(x)
            h_t = s.output()

            ctx_t, alpha_t = self.attend(H_source, s, w1dt, h_len, curr_bsize)
            output_vector = w * dy.concatenate([h_t, ctx_t]) + b
            #probs = dy.softmax(output_vector)
            ctx_t0 = ctx_t
            if dropout_config:
                output_vector = dy.dropout(output_vector, dropout_val)

            temp_loss = dy.pickneglogsoftmax_batch(output_vector, dec_wrds[sent])

            if 0 in dec_mask[sent]:
                mask_expr = dy.inputVector(dec_mask[sent])
                mask_expr = dy.reshape(mask_expr, (1, ), curr_bsize)
                temp_loss = temp_loss * mask_expr

            loss.append(temp_loss)

        loss = dy.esum(loss)
        loss = dy.sum_batches(loss) / batch_size
        return loss

    def attend(self, input_mat, state, w1dt, h_len, curr_bsize):
        global attention_w2
        global attention_v
        w2 = dy.parameter(attention_w2)
        v = dy.parameter(attention_v)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim x attdim)
        w2dt = w2*dy.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector

        unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
        att_weights = dy.reshape(unnormalized, (h_len, ), curr_bsize)
        att_weights = dy.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context, att_weights

    def generate(self, in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
        embedded = self.embed_sentence(in_seq, True)
        encoded = self.encode_generation(enc_fwd_lstm, enc_bwd_lstm, embedded)
        h_len = len(encoded)
        curr_bsize = 1

        w = dy.parameter(decoder_w)
        b = dy.parameter(decoder_b)
        w1 = dy.parameter(attention_w1)

        H_source = dy.concatenate_cols(encoded)
        s = dec_lstm.initial_state()
        ctx_t0 = dy.vecInput(hidden_size * 2)
        last_output_embeddings = output_lookup[word2idx_en['<s>']]
        w1dt = w1 * H_source

        out = []
        count_EOS = 0
        for i in range(len(in_seq)*2):
            if count_EOS == 1: break
            # w1dt can be computed and cached once for the entire decoding phase
            x = dy.concatenate([ctx_t0, last_output_embeddings])
            #print "Attention: Generate"
            s = s.add_input(x)
            h_t = s.output()
            ctx_t, alpha_t = self.attend(H_source, s, w1dt, h_len, curr_bsize)

            out_vector = w * dy.concatenate([h_t, ctx_t]) + b
            probs = dy.softmax(out_vector).vec_value()
            next_char = probs.index(max(probs))
            last_output_embeddings = output_lookup[next_char]
            if idx2word_en[next_char] == '<EOS>':
                count_EOS += 1
                continue

            out.append(idx2word_en[next_char])
            ctx_t0 = ctx_t

        return ' '.join(out)

    def calculate_bleu_score(self):
        print "generating sentences"

        with open("gen_e1k_18.txt", "w") as out:
            for i, sent in enumerate(encoder_test_wids):
                dy.renew_cg()
                decoded_test_sent = self.generate(sent, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
                out.write(decoded_test_sent + "\n")

    def train(self, train_set_de, train_set_en, valid_enc, valid_dec):
        train_set = [[i, j] for i, j in zip(train_set_de, train_set_en)]
        train_set.sort(key=lambda item: (-len(item[0]), item))

        valid_set = [[i, j] for i, j in zip(valid_enc, valid_dec)]
        valid_set.sort(key=lambda item: (-len(item[0]), item))

        minibatches = self.get_batch(train_set, batch_size)
        valid_minibatches = self.get_batch(valid_set, valid_batch_size)

        trainer = dy.SimpleSGDTrainer(model)

        for i in range(no_epochs):
            random.shuffle(minibatches)
            random.shuffle(valid_minibatches)

            for j, mini in enumerate(minibatches):
                mini = zip(*mini)

                enc_batch, dec_batch = list(mini[0]), list(mini[1])
                curr_bsize = len(enc_batch)
                loss = self.get_decoded(enc_batch, dec_batch, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
                loss_value = loss.value()
                loss.backward()
                trainer.update()

                perplexity = np.exp(float(loss_value * curr_bsize) / sum(len(s) for s in dec_batch))

                if j%100 == 0:
                    total_vloss = 0
                    total_toks = 0
                    for k in valid_minibatches:
                        valid_mini = zip(*k)
                        valid_enc, valid_dec = list(valid_mini[0]), list(valid_mini[1])
                        cbsize = len(valid_enc)
                        valid_loss = self.get_decoded(valid_enc, valid_dec, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
                        vloss_val = valid_loss.value()
                        total_vloss = (vloss_val * cbsize) + total_vloss

                    valid_word_perp = np.exp(total_vloss/float(total_vtoks))

                    with codecs.open('drope_debug.txt', 'a+') as out:
                        out.write("Epoch: " + str(i) + "   Samples:" + str(j) + "   LOSS: " + str(loss_value) + " Perplexity: " +  str(perplexity) + "\n")
                        out.write("Epoch: " + str(i) + "   Samples:" + str(j) +  " Valid_Loss: " + str(total_vloss) + " Valid_Perplexity: " + str(valid_word_perp) + "\n")
                        out.write(str(self.generate(enc_batch[-1], enc_fwd_lstm, enc_bwd_lstm, dec_lstm)) + "\n")
            model.save('dropmodel_'+str(i) + '_' + str(j), [enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, output_lookup,
                            attention_w1, attention_w2, attention_v, decoder_w, decoder_b])

net = AttentionModel()
if TRAIN_SWITCH:
    net.train(data_de, data_en, valid_enc_ids, valid_dec)
else:
    net.calculate_bleu_score()
