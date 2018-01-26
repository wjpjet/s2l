import numpy as np

class Vocabulary(object):
    def __init__(self, labels_file, itos):
        self.embeddings = None
        self.stoi = None
        self.itos = itos

        # Load labels
        with open(labels_file,'r') as f:
            label_id = eval(f.read())
        self.label_id = label_id
        self.num_classes = len(label_id)
    
    def set_word_embedding(self, embedding_file, vocab_file, embed_d=300):        
            
        # Read in all words from vocab file
        with open(vocab_file, 'r') as f:
            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        
        # Set int to string array as words from vocab file
        self.itos = words
        
        # Build stoi (string to int) from loaded words list
        stoi = {w: idx for idx, w in enumerate(words)}
        self.stoi = stoi
    
        # read in all values from glove embedding
        with open(embedding_file, 'r') as f:
            vectors = {}
            for line in f:
                vals = line.rstrip().split(' ')
                if vals[0] in words:
                    vectors[vals[0]] = [float(x) for x in vals[1:]]
        
        # Initialize empty word embedding as np array
        vocab_size = len(words)
        W = np.zeros((vocab_size, embed_d), dtype=np.float32)
        
        for i, word in enumerate(self.itos):
            #if word == '<unk>':
            #    np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
            try:
                W[i, :] = vectors[word]
            except: # If word isn't in vocab, initialize random array
                W[i, :] = np.random.normal( embed_d )
        
        W = W.astype(np.float32)
        # Assign numpy array as W
        self.embeddings = W

    def dummy_embedding(self, embed_d=300):
        # Dummy embedding using random initialization.
        for i, w in enumerate(self.stoi):
            self.itos[w] = i
        vocab_size = len(self.stoi)
        self.embeddings = np.asarray(np.random.randn(vocab_size, embed_d), dtype=np.float32)
        
    def process_data(self, data):
        numerialized_data = []
        for s1, s2, y in data:
            
            n_s1 = []
            n_s2 = []
            n_s1.append(self.stoi['<sos>'])
            n_s2.append(self.stoi['<sos>'])
            for word in s1:
                if word in self.stoi:
                    n_s1.append(self.stoi[word])
                else:
                    n_s1.append(self.stoi['<unk-0>'])
            for word in s2:
                if word in self.stoi:
                    n_s2.append(self.stoi[word])
                else:
                    n_s2.append(self.stoi['<unk-0>'])
            
            n_s1.append(self.stoi['<eos>'])
            n_s2.append(self.stoi['<eos>'])

            n_y = self.label_id[y]
            
            numerialized_data.append((n_s1, n_s2, n_y))
            
        return numerialized_data