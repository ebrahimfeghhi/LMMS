# import mkl
# mkl.set_num_threads(8)

from collections import defaultdict
import numpy as np


def get_sk_type(sensekey):
    return int(sensekey.split('%')[1].split(':')[0])


def get_sk_pos(sk, tagtype='long'):
    # merges ADJ with ADJ_SAT

    if tagtype == 'long':
        type2pos = {1: 'NOUN', 2: 'VERB', 3: 'ADJ', 4: 'ADV', 5: 'ADJ'}
        return type2pos[get_sk_type(sk)]

    elif tagtype == 'short':
        type2pos = {1: 'n', 2: 'v', 3: 's', 4: 'r', 5: 's'}
        return type2pos[get_sk_type(sk)]


def get_sk_lemma(sensekey):
    return sensekey.split('%')[0]


class SensesVSM(object):

    def __init__(self, vecs_path, normalize=True):
        self.vecs_path = vecs_path
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        if self.vecs_path.endswith('.txt'):
            self.load_txt(self.vecs_path)

        elif self.vecs_path.endswith('.npz'):
            self.load_npz(self.vecs_path)

        self.load_aux_senses()

        if normalize:
            self.normalize()

    def load_txt(self, txt_vecs_path):
        self.vectors = []
        with open(txt_vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))
        self.vectors = np.vstack(self.vectors)

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def load_npz(self, npz_vecs_path):
        loader = np.load(npz_vecs_path)
        self.labels = loader['labels'].tolist()
        self.vectors = loader['vectors']

        self.labels_set = set(self.labels)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def load_aux_senses(self):
        self.sk_lemmas = {sk: get_sk_lemma(sk) for sk in self.labels}
        self.sk_postags = {sk: get_sk_pos(sk) for sk in self.labels}

        self.lemma_sks = defaultdict(list)
        for sk, lemma in self.sk_lemmas.items():
            self.lemma_sks[lemma].append(sk)
        self.known_lemmas = set(self.lemma_sks.keys())

        self.sks_by_pos = defaultdict(list)
        for s in self.labels:
            self.sks_by_pos[self.sk_postags[s]].append(s)
        self.known_postags = set(self.sks_by_pos.keys())
        
        # create a dictionary where the keys are word_pos
        # and the values are the sensekeys for that word/pos
        # combination.
        self.word_pos_sk= defaultdict(lambda: defaultdict(list))
        for sk in self.labels:

            # extract word from sensekey
            lemma = get_sk_lemma(sk)
            pos = get_sk_pos(sk)
            
            # add sensekey, lowercase lemma for easier search
            self.word_pos_sk[lemma.lower()][pos].append(sk)
    
    def save_npz(self):
        npz_path = self.vecs_path.replace('.txt', '.npz')
        np.savez_compressed(npz_path,
                            labels=self.labels,
                            vectors=self.vectors)

    def normalize(self, norm='l2'):
        norms = np.linalg.norm(self.vectors, axis=1)
        self.vectors = (self.vectors.T / norms).T

    def get_vec(self, label):
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()
    
    def num_senses(self, lemma, postag_list):

        relevant_sks = []
        breakpoint()
        for sk in self.labels:
            if (lemma is None) or (self.sk_lemmas[sk] == lemma) or ((self.sk_lemmas[sk] == lemma.lower())):
                if (postag_list is None) or (self.sk_postags[sk] in postag_list):
                    relevant_sks.append(sk)


        return len(relevant_sks)
    
    def num_senses_fast(self, lemma_arr, postag_arr):
        
        num_senses_dict = {}
        for i, (lemma, postag) in enumerate(zip(lemma_arr, postag_arr)):
            
            # find the number of senses for that lemma + postag
            num_senses_dict[f'{lemma}_{postag}'] = len(self.word_pos_sk[lemma.lower()][postag])
        
        return num_senses_dict
    
    def get_sense_embeddings_sentence(self, lemma_arr, postag_arr, ctx_embeddings):
   
        
        sense_embeddings = []
        no_sense_embedding_words = []
        
        for i, (lemma, postag, ctx_embed) in enumerate(zip(lemma_arr, postag_arr, ctx_embeddings)):
            
            # get relevant sense embeddings
            relevant_sks = self.word_pos_sk[lemma.lower()][postag]
            
            # if none, add to this list so that we can get GloVe embeddings
            if len(relevant_sks) == 0:
                no_sense_embedding_words.append(lemma)
                continue
                
            
            # get indices for each sensekey
            relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]
            
            # use indices to get sense embeddings, compute cosine sim with contextual embed 
            norm_ctx_embed = ctx_embed[1] / np.linalg.norm(ctx_embed[1])
            sims = np.dot(self.vectors[relevant_sks_idxs], norm_ctx_embed)
            
            #matches = list(zip(relevant_sks, sims))
            #matches = sorted(matches, key=lambda x: x[1], reverse=True)
            #sims_indices_sorted = np.argsort(sims)
            top_matching_idx = np.argmax(sims)
            top_1_sense_vector = self.vectors[relevant_sks_idxs][top_matching_idx]
            sense_embeddings.append(top_1_sense_vector) 
        
        return sense_embeddings, no_sense_embedding_words
    
    
    def match_senses(self, vec, lemma=None, postag=None, topn=100):
        
        relevant_sks = []
        
        # WSD
        if (lemma is not None) and (postag is not None):
            for sk in self.labels:
                if self.sk_lemmas[sk] == lemma:
                        if self.sk_postags[sk] == postag:
                            relevant_sks.append(sk)
            relevant_sks_idxs = [self.indices[sk] for sk in relevant_sks]
            
        # USM                
        else:
            relevant_sks_idxs = self.indices
            
        if len(relevant_sks) == 0:
            return None, None
      
        sims = np.dot(self.vectors[relevant_sks_idxs], np.array(vec))
        matches = list(zip(relevant_sks, sims))

        matches = sorted(matches, key=lambda x: x[1], reverse=True)
        
        sims_indices_sorted = np.argsort(sims)
        top_1_sense_vector = self.vectors[relevant_sks_idxs][sims_indices_sorted[-1]]
        
        return matches[:topn], top_1_sense_vector
    
    def match_senses_usm(self, ctx_embedding_mat, topn, average=False):
        
        '''
        :param embedding_mat: matrix with contextual embeddings 
        :param int topn: collect top N sense embeddings (based on cosine sim)
        :param bool average: If true, take mean across top N sense embeddings 
        with the weights being the softmax of the cosine sim values 
        
        Outputs the closest/average of closest matching sense embeddings for each contextual embedding.
        Does not restrict sense embeddings based on word or POS.
        '''
        
        sims = self.vectors@ctx_embedding_mat
        sorted_idxs_topN = np.argsort(sims, axis=0)[-topn:, :]
        topN_vecs = self.vectors[sorted_idxs_topN, :]
        
        if average:
            raise NotImplementedError()
        else:
            # if not averaging, just pick the one with the highest cosine simalarity 
            return topN_vecs[-1]
        
    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.vectors, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()


class VSM(object):

    # def __init__(self, vecs_path, normalize=True):
    def __init__(self):
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0

        # self.load_txt(vecs_path)

        # if normalize:
        #     self.normalize()

    def load(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]
        
    def load_txt(self, vecs_path):
        self.vectors = []
        with open(vecs_path, encoding='utf-8') as vecs_f:
            for line_idx, line in enumerate(vecs_f):
                elems = line.split()
                self.labels.append(elems[0])
                self.vectors.append(np.array(list(map(float, elems[1:])), dtype=np.float32))

        self.labels_set = set(self.labels)
        self.vectors = np.vstack(self.vectors)
        self.indices = {l: i for i, l in enumerate(self.labels)}
        self.ndims = self.vectors.shape[1]

    def normalize(self, norm='l2'):
        self.vectors = (self.vectors.T / np.linalg.norm(self.vectors, axis=1)).T

    def get_vec(self, label):
        return self.vectors[self.indices[label]]

    def similarity(self, label1, label2):
        v1 = self.get_vec(label1)
        v2 = self.get_vec(label2)
        return np.dot(v1, v2).tolist()

    def most_similar_vec(self, vec, topn=10):
        sims = np.dot(self.vectors, vec).astype(np.float32)
        sims_ = sims.tolist()
        r = []
        for top_i in sims.argsort().tolist()[::-1][:topn]:
            r.append((self.labels[top_i], sims_[top_i]))
        return r

    def most_similar(self, label, topn=10):
        return self.most_similar_vec(self.get_vec(label), topn=topn)

    def sims(self, vec):
        return np.dot(self.vectors, np.array(vec)).tolist()


if __name__ == '__main__':

    vecs_path = 'data/vectors/bert-large-cased/lmms-sp-wsd.bert-large-cased.vectors.txt'
    vsm = SensesVSM(vecs_path)
    print(len(np.unique(vsm.vectors, axis=0)))
