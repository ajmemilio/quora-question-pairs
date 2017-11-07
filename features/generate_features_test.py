import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag 
from string import punctuation
from collections import Counter

print "Started"
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def cleaning_text(text, remove_stopwords=False, stem_words=False):
    # Convert words to lower case and split them
    # ATTENTION over cleaning
    #text = text.lower().split()
    text = text.split()

    # Optionally, remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        #text = [w for w in text if not w in stops]
        text = [w for w in text if not w.lower() in stops]
    
    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"what's", "", text)
    text = re.sub(r"What's", "", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r" m ", " am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"60k", " 60000 ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e-mail", "email", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"quikly", "quickly", text)
    text = re.sub(r" usa ", " America ", text)
    text = re.sub(r" USA ", " America ", text)
    text = re.sub(r" u s ", " America ", text)
    text = re.sub(r" uk ", " England ", text)
    text = re.sub(r" UK ", " England ", text)
    text = re.sub(r"india", "India", text)
    text = re.sub(r"switzerland", "Switzerland", text)
    text = re.sub(r"china", "China", text)
    text = re.sub(r"chinese", "Chinese", text) 
    text = re.sub(r"imrovement", "improvement", text)
    text = re.sub(r"intially", "initially", text)
    text = re.sub(r"quora", "Quora", text)
    text = re.sub(r" dms ", "direct messages ", text)  
    text = re.sub(r"demonitization", "demonetization", text) 
    text = re.sub(r"actived", "active", text)
    text = re.sub(r"kms", " kilometers ", text)
    text = re.sub(r"KMs", " kilometers ", text)
    text = re.sub(r" cs ", " computer science ", text) 
    text = re.sub(r" upvotes ", " up votes ", text)
    text = re.sub(r" iPhone ", " phone ", text)
    text = re.sub(r"\0rs ", " rs ", text) 
    text = re.sub(r"calender", "calendar", text)
    text = re.sub(r"ios", "operating system", text)
    text = re.sub(r"gps", "GPS", text)
    text = re.sub(r"gst", "GST", text)
    text = re.sub(r"programing", "programming", text)
    text = re.sub(r"bestfriend", "best friend", text)
    text = re.sub(r"dna", "DNA", text)
    text = re.sub(r"III", "3", text) 
    text = re.sub(r"the US", "America", text)
    text = re.sub(r"Astrology", "astrology", text)
    text = re.sub(r"Method", "method", text)
    text = re.sub(r"Find", "find", text) 
    text = re.sub(r"banglore", "Banglore", text)
    text = re.sub(r" J K ", " JK ", text)

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)
    
    # Return a list of words
    return(text)

def word_match_share(questions):
    q1words = set(questions["question1"].lower().split())
    q2words = set(questions["question2"].lower().split())

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.0

    shared_words = q1words.intersection(q2words)
    #count share
    shared_rate = len(shared_words) / float(len(q1words) + len(q2words) - len(shared_words))
    return shared_rate

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller
def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0.0
    else:
        return 1.0 / (count + eps)

def tfidf_word_match_share(questions, weights):
    q1words = questions['question1'].lower().split()
    q2words = questions['question2'].lower().split()

    if len(q1words) == 0 or len(q2words) == 0:
       # The computer-generated chaff includes a few questions that are nothing but stopwords
       return 0.0

    shared_weights_q1 = [weights.get(w, 0) for w in iter(q1words) if w in q2words]
    shared_weights_q2 = [weights.get(w, 0) for w in iter(q2words) if w in q1words]
    
    total_weights = [weights.get(w, 0) for w in iter(q1words)] + [weights.get(w, 0) for w in iter(q2words)]
    
    sum_total_weights = float(np.sum(total_weights))
    sum_shared_weights = float(np.sum(shared_weights_q1) + np.sum(shared_weights_q2))

    if sum_total_weights == 0.0:
       return 0.0

    tfidf_match_share = sum_shared_weights / sum_total_weights

    return tfidf_match_share

def pos_tagger(questions, tag):
    q1words = word_tokenize(questions['question1'])
    q2words = word_tokenize(questions['question2'])

    q1tagged = pos_tag(q1words)
    q2tagged = pos_tag(q2words)

    tagged = 0.0

    q1list = [word for word, t in q1tagged if t == tag]
    q2list = [word for word, t in q2tagged if t == tag]

    if len(q1list) != 0 and len(q2list) != 0:
        count = len(q1list) + len(q2list)
        count_shared_q1 = len([word for word in q1list if word in q2list])
        count_shared_q2 = len([word for word in q2list if word in q1list])
        tagged = float(count_shared_q1 + count_shared_q2) / float(count)
    
    return tagged

def pos_tagger_len_q1(question, tag):
    q1words = word_tokenize(question)
    q1tagged = pos_tag(q1words)
    q1list = [word for word, t in q1tagged if t == tag]
    return len(q1list)

def pos_tagger_len_q2(question, tag):
    q2words = word_tokenize(question)
    q2tagged = pos_tag(q2words)
    q2list = [word for word, t in q2tagged if t == tag]
    return len(q2list)    

def words_r2gram(questions):
    q1words = questions["question1"].lower().split()
    q2words = questions["question2"].lower().split()

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.0

    q1_2gram = set([i for i in zip(q1words, q1words[1:])])
    q2_2gram = set([i for i in zip(q2words, q2words[1:])])
    shared_2gram = q1_2gram.intersection(q2_2gram)

    if len(q1_2gram) + len(q2_2gram) == 0:
        r2gram_rate = 0.0
    else:
        r2gram_rate = len(shared_2gram) / float(len(q1_2gram) + len(q2_2gram) - len(shared_2gram))

    return r2gram_rate

def stemming_shared(questions):
    q1words = questions["question1"].lower().split()
    q2words = questions["question2"].lower().split()

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.0

    stemmer = SnowballStemmer('english')
    q1stemmed_words = [stemmer.stem(word) for word in q1words]
    q2stemmed_words = [stemmer.stem(word) for word in q2words]

    q1stemmed_words = set(q1stemmed_words)
    q2stemmed_words = set(q2stemmed_words)

    shared_stemmed_words = q1stemmed_words.intersection(q2stemmed_words)    
    shared_stemmed_words_rate = len(shared_stemmed_words) / float(len(q1stemmed_words) + len(q2stemmed_words) - len(shared_stemmed_words))

    return shared_stemmed_words_rate

def words_hamming(questions):
    q1words = questions["question1"].lower().split()
    q2words = questions["question2"].lower().split()

    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.0

    words_hamming = sum(1 for i in zip(q1words, q2words) if i[0]==i[1])/ float(max(len(q1words), len(q2words)))

    return words_hamming

def jaccard(row):
    wic = set(row['question1']).intersection(set(row['question2']))
    uw = set(row['question1']).union(row['question2'])
    if len(uw) == 0:
        uw = [1]
    return float(len(wic)) / float(len(uw))

def common_words(row):
    return len(set(row['question1']).intersection(set(row['question2'])))

def total_unique_words(row):
    return len(set(row['question1']).union(row['question2']))

def wc_diff(row):
    return abs(len(row['question1']) - len(row['question2']))

def wc_ratio(row):
    l1 = float(len(row['question1']))
    l2 = float(len(row['question2']))
    if l2 == 0.0:
        return np.nan
    if l1 >= l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['question1'])) - len(set(row['question2'])))

def wc_ratio_unique(row):
    l1 = float(len(set(row['question1'])))
    l2 = float(len(set(row['question2'])))
    if l2 == 0.0:
        return np.nan
    if l1 >= l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['question1'] or not row['question2']:
        return np.nan
    return int(row['question1'][0] == row['question2'][0])

def char_diff(row):
    return abs(len(''.join(row['question1'])) - len(''.join(row['question2'])))

def char_ratio(row):
    l1 = float(len(''.join(row['question1'])))
    l2 = float(len(''.join(row['question2'])))
    if l2 == 0.0:
        return np.nan
    if l1 >= l2:#verificar erro
        return l2 / l1
    else:
        return l1 / l2

def main():
    # Add the string 'empty' to empty strings
    print "Remove empty values"
    df_train = train.fillna('empty')
    df_test = test.fillna('empty')
    
    print "Calculate weights TFIDF"
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    #test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
    eps = 5000
    words = (" ".join(train_qs)).split()
    counts = Counter(words)
    weights = {word: get_weight(count, eps) for word, count in counts.items()}
    
    print "Clean data test"
    df_test['question1'] = df_test['question1'].apply(lambda q: cleaning_text(q, remove_stopwords=True, stem_words=False))
    df_test['question2'] = df_test['question2'].apply(lambda q: cleaning_text(q, remove_stopwords=True, stem_words=False))

    print "pos tag len VB_q1 test"
    df_test['len_vb_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'VB'))
    print "pos tag len VBD_q1 test"
    df_test['len_vbd_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'VBD'))
    print "pos tag len VBG_q1 test"
    df_test['len_vbg_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'VBG'))
    print "pos tag len VBN_q1 test"
    df_test['len_vbn_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'VBN'))
    print "pos tag len VBP_q1 test"
    df_test['len_vbp_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'VBP'))
    print "pos tag len VBZ_q1 test"
    df_test['len_vbz_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'VBZ'))
    print "pos tag len NN_q1 test"
    df_test['len_nn_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'NN'))
    print "pos tag len NNP_q1 test"
    df_test['len_nnp_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'NNP'))
    print "pos tag len NNPS_q1 test"
    df_test['len_nnps_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'NNPS'))
    print "pos tag len JJ_q1 test"
    df_test['len_jj_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'JJ'))
    print "pos tag len JJR_q1 test"
    df_test['len_jjr_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'JJR'))
    print "pos tag len JJS_q1 test"
    df_test['len_jjs_q1'] = df_test['question1'].apply(lambda qs: pos_tagger_len_q1(qs, 'JJS'))

    print "pos tag len VB_q2 test"
    df_test['len_vb_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'VB'))
    print "pos tag len VBD_q2 test"
    df_test['len_vbd_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'VBD'))
    print "pos tag len VBG_q2 test"
    df_test['len_vbg_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'VBG'))
    print "pos tag len VBN_q2 test"
    df_test['len_vbn_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'VBN'))
    print "pos tag len VBP_q2 test"
    df_test['len_vbp_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'VBP'))
    print "pos tag len VBZ_q2 test"
    df_test['len_vbz_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'VBZ'))
    print "pos tag len NN_q2 test"
    df_test['len_nn_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'NN'))
    print "pos tag len NNP_q2 test"
    df_test['len_nnp_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'NNP'))
    print "pos tag len NNPS_q2 test"
    df_test['len_nnps_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'NNPS'))
    print "pos tag len JJ_q2 test"
    df_test['len_jj_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'JJ'))
    print "pos tag len JJR_q2 test"
    df_test['len_jjr_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'JJR'))
    print "pos tag len JJS_q2 test"
    df_test['len_jjs_q2'] = df_test['question2'].apply(lambda qs: pos_tagger_len_q2(qs, 'JJS'))
    
    print "Diff pos tag len VB test"
    df_test['diff_vb'] = df_test['len_vb_q1'] - df_test['len_vb_q2']
    print "Diff pos tag len VBD test"
    df_test['diff_vbd'] = df_test['len_vbd_q1'] - df_test['len_vbd_q2']
    print "Diff pos tag len VBG test"
    df_test['diff_vbg'] = df_test['len_vbg_q1'] - df_test['len_vbg_q2']
    print "Diff pos tag len VBN test"
    df_test['diff_vbn'] = df_test['len_vbn_q1'] - df_test['len_vbn_q2']
    print "Diff pos tag len VBP test"
    df_test['diff_vbp'] = df_test['len_vbp_q1'] - df_test['len_vbp_q2']
    print "Diff pos tag len VBZ test"
    df_test['diff_vbz'] = df_test['len_vbz_q1'] - df_test['len_vbz_q2']
    print "Diff pos tag len NN test"
    df_test['diff_nn'] = df_test['len_nn_q1'] - df_test['len_nn_q2']
    print "Diff pos tag len NNP test"
    df_test['diff_nnp'] = df_test['len_nnp_q1'] - df_test['len_nnp_q2']
    print "Diff pos tag len NNPS test"
    df_test['diff_nnps'] = df_test['len_nnps_q1'] - df_test['len_nnps_q2']
    print "Diff pos tag len JJ test"
    df_test['diff_jj'] = df_test['len_jj_q1'] - df_test['len_jj_q2']
    print "Diff pos tag len JJR test"
    df_test['diff_jjr'] = df_test['len_jjr_q1'] - df_test['len_jjr_q2']
    print "Diff pos tag len JJS test"
    df_test['diff_jjs'] = df_test['len_jjs_q1'] - df_test['len_jjs_q2']
    
    print "Pos tagger VB test"
    df_test['VB'] = df_test.apply(lambda qs: pos_tagger(qs, 'VB'), axis=1, raw=True)
    print "Pos tagger VBD test"
    df_test['VBD'] = df_test.apply(lambda qs: pos_tagger(qs, 'VBD'), axis=1, raw=True)
    print "Pos tagger VBG test"
    df_test['VBG'] = df_test.apply(lambda qs: pos_tagger(qs, 'VBG'), axis=1, raw=True)
    print "Pos tagger VBN test"
    df_test['VBN'] = df_test.apply(lambda qs: pos_tagger(qs, 'VBN'), axis=1, raw=True)
    print "Pos tagger VBP test"
    df_test['VBP'] = df_test.apply(lambda qs: pos_tagger(qs, 'VBP'), axis=1, raw=True)
    print "Pos tagger VBZ test"
    df_test['VBZ'] = df_test.apply(lambda qs: pos_tagger(qs, 'VBZ'), axis=1, raw=True)
    print "Pos tagger NN test"
    df_test['NN'] = df_test.apply(lambda qs: pos_tagger(qs, 'NN'), axis=1, raw=True)
    print "Pos tagger NNP test"
    df_test['NNP'] = df_test.apply(lambda qs: pos_tagger(qs, 'NNP'), axis=1, raw=True)
    print "Pos tagger NNPS test"
    df_test['NNPS'] = df_test.apply(lambda qs: pos_tagger(qs, 'NNPS'), axis=1, raw=True)
    print "Pos tagger JJ test"
    df_test['JJ'] = df_test.apply(lambda qs: pos_tagger(qs, 'JJ'), axis=1, raw=True)
    print "Pos tagger JJR test"
    df_test['JJR'] = df_test.apply(lambda qs: pos_tagger(qs, 'JJR'), axis=1, raw=True)
    print "Pos tagger JJS test"
    df_test['JJS'] = df_test.apply(lambda qs: pos_tagger(qs, 'JJS'), axis=1, raw=True)
    
    print "Word_match_share test"
    df_test['word_match_share'] = df_test.apply(word_match_share, axis=1, raw=True)

    print "Len question1 test"
    df_test['len_q1'] = df_test['question1'].apply(lambda x: len(str(x)))

    print "Len question2 test"
    df_test['len_q2'] = df_test['question2'].apply(lambda x: len(str(x)))

    print "Diff questions test"
    df_test['diff_len'] = df_test['len_q1'] - df_test['len_q2']

    print "Words r2gram test"
    df_test['words_r2gram'] = df_test.apply(words_r2gram, axis=1, raw=True)

    print "Stemming shared test"
    df_test['stemming_shared'] = df_test.apply(stemming_shared, axis=1, raw=True)

    print "Words hamming test"
    df_test['words_hamming'] = df_test.apply(words_hamming, axis=1, raw=True)

    print "Tfidf_word_match_share test"
    df_test['tfidf_word_match_share'] = df_test.apply(lambda qs: tfidf_word_match_share(qs, weights), axis=1, raw=True)

    print "jaccard"
    df_test['jaccard'] = df_test.apply(jaccard, axis=1, raw=True) 
    
    print "wc_diff"
    df_test['wc_diff'] = df_test.apply(wc_diff, axis=1, raw=True) 
    
    print "wc_ratio"
    df_test['wc_ratio'] = df_test.apply(wc_ratio, axis=1, raw=True) 
    
    print "wc_diff_unique"
    df_test['wc_diff_unique'] = df_test.apply(wc_diff_unique, axis=1, raw=True) 
    
    print "wc_ratio_unique"
    df_test['wc_ratio_unique'] = df_test.apply(wc_ratio_unique, axis=1, raw=True) 

    print "same_start"
    df_test['same_start'] = df_test.apply(same_start_word, axis=1, raw=True) 
    
    print "char_diff"
    df_test['char_diff'] = df_test.apply(char_diff, axis=1, raw=True) 

    print "total_unique_words"
    df_test['total_unique_words'] = df_test.apply(total_unique_words, axis=1, raw=True)
    
    print "char_ratio"
    df_test['char_ratio'] = df_test.apply(char_ratio, axis=1, raw=True)  

    #to csv test
    df_test.to_csv('test_v62.csv', index=False)
    
    
main()
print "Done"