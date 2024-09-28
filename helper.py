from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
import re
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
import contractions
import distance
import pickle
import numpy as np

sw_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
cv = pickle.load(open('cv.pkl','rb'))


def preprocess(row):
    ps = PorterStemmer()
    row = str(row).lower().strip()

    row = row.replace("%", "percent")
    row = row.replace("$", "dollar")
    row = row.replace('₹', "rupee")
    row = row.replace('€', "euro")
    row = row.replace("@", "at")

    row = row.replace("[math]", " ")

    # Regex Tutorial at:https://www3.ntu.edu.sg/home/ehchua/programming/howto/Regexe.html
    # Using Contractions Library else list available at:https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/19794953#19794953

    row = re.sub(r"([0-9]+)000000000", r"\1b", row)
    row = re.sub(r"([0-9]+)000000", r"\1m", row)
    row = re.sub(r"([0-9]+)000", r"\1k", row)

    row = contractions.fix(row)

    row = BeautifulSoup(row,"html.parser")
    row = row.get_text()

    ptrn = re.compile("\W")
    row = re.sub(ptrn, " ", row).strip()

    row = " ".join([ps.stem(word) for word in row.split(" ")])

    return row


def common_words(q1,q2):
    ques1=set(q1.lower().strip().split(" "))
    ques2=set(q2.lower().strip().split(" "))
    return len(ques1&ques2)


def total_words(q1,q2):
    ques1=set(q1.lower().strip().split(" "))
    ques2=set(q2.lower().strip().split(" "))
    return len(ques1)+len(ques2)


def get_token_features(q1, q2):
    ques1 = q1
    ques2 = q2

    ens = 10 ** -6

    token_features = [0.0] * 8

    ques1_tokens = ques1.split(" ")
    ques2_tokens = ques2.split(" ")

    if len(ques1_tokens) == 0 or len(ques2_tokens) == 0:
        return token_features

    ques1_words = set([word for word in ques1_tokens if word not in sw_list])
    ques2_words = set([word for word in ques2_tokens if word not in sw_list])

    ques1_stopwords = set([word for word in ques1_tokens if word in sw_list])
    ques2_stopwords = set([word for word in ques2_tokens if word in sw_list])

    common_stopwords = len(ques1_stopwords & ques2_stopwords)
    common_words = len(ques1_words & ques2_words)
    common_tokens = len(set(ques1_tokens) & set(ques2_tokens))

    token_features[0] = common_words / (min(len(ques1_words), len(ques2_words)) + ens)
    token_features[1] = common_words / (max(len(ques1_words), len(ques2_words)) + ens)
    token_features[2] = common_stopwords / (min(len(ques1_stopwords), len(ques2_stopwords)) + ens)
    token_features[3] = common_stopwords / (max(len(ques1_stopwords), len(ques2_stopwords)) + ens)
    token_features[4] = common_tokens / (min(len(ques1_tokens), len(ques2_tokens)) + ens)
    token_features[5] = common_tokens / (max(len(ques1_tokens), len(ques2_tokens)) + ens)

    token_features[6] = int(ques1_tokens[-1] == ques2_tokens[-1])
    token_features[7] = int(ques1_tokens[0] == ques2_tokens[0])

    return token_features


def get_length_features(q1, q2):
    ques1 = q1
    ques2 = q2

    length_features = [0.0] * 3

    ques1_tokens = ques1.split(" ")
    ques2_tokens = ques2.split(" ")

    if len(ques1_tokens) == 0 or len(ques2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(ques1_tokens) - len(ques2_tokens))
    length_features[1] = (len(ques1_tokens) + len(ques2_tokens)) / 2

    comm = distance.lcsubstrings(ques1, ques2)
    comm = list(comm)
    val = len(comm[0]) if comm else 0
    length_features[2] = val / (min(len(ques1), len(ques2)) + 10 ** -6)

    return length_features


def get_fuzzy_features(q1, q2):
    ques1 = q1
    ques2 = q2

    fuzzy_features = [0.0] * 4

    fuzzy_features[0] = fuzz.ratio(ques1, ques2)

    fuzzy_features[1] = fuzz.partial_ratio(ques1, ques2)

    fuzzy_features[2] = fuzz.token_sort_ratio(ques1, ques2)

    fuzzy_features[3] = fuzz.token_set_ratio(ques1, ques2)

    return fuzzy_features


def query_creator(q1, q2):
    input_query = []

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(common_words(q1, q2))
    input_query.append(total_words(q1, q2))
    input_query.append(round(common_words(q1, q2) / total_words(q1, q2), 2))

    token_features = get_token_features(q1, q2)
    input_query.extend(token_features)

    length_features = get_length_features(q1, q2)
    input_query.extend(length_features)

    fuzzy_features = get_fuzzy_features(q1, q2)
    input_query.extend(fuzzy_features)

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))