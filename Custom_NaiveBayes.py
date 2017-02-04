from collections import Counter, defaultdict

import math, random, re, glob

def tokenize(message):
    message = message.lower()
    all_words = re.findall('[a-z0-9]', message)
    lowercase =  set(all_words)
    return lowercase

def count_words(train):
    count= defaultdict(lambda: [0,0])
    for message, is_spam in train:
        for words in tokenize(message):
            count[words][0 if is_spam else 1] +=1
    return count

def word_prob(count,spams,non_spams, k = 0.5):
    return [(w, (spam + k)/(spams + 2*k), (non_spam + k)/(non_spams + 2*k)) for w,(spam,non_spam) in count.items()]

def spam_prob(word_probs, message):
    words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0
    for word,prob_if_spam, prob_if_not_spam in word_probs:
        if word in words:
            log_prob_if_spam +=math.log(prob_if_spam)
            log_prob_if_not_spam+=math.log(prob_if_not_spam)
        else:
            log_prob_if_spam +=math.log(1.0-prob_if_spam)
            log_prob_if_not_spam +=math.log(1.0 - prob_if_not_spam)

    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam/(prob_if_spam + prob_if_not_spam)

class NBClassifier:
    def __init__(self, k = 0.5):
        self.k = k
        self.word_probs = []

    def train(self, train_data):
        num_spams = len([is_spam for message, is_spam in train_data if is_spam])
        num_non_spams = len(train_data) - num_spams
        word_count = count_words(train_data)
        self.word_probs = word_prob(word_count,num_spams, num_non_spams, self.k)

    def classify(self, message):
        return spam_prob(self.word_probs, message)


path = "/home/beautifulsoup4/PycharmProjects/untitled4/*/*"
data = []
subject_regex = re.compile(r"^Subject:\s+")
for fn in glob.glob(path):
    is_spam = "ham" in fn

    with open(fn,'r', encoding='ISO-8859-1') as file:
        for line in file:
            if line.startswith("Subject:"):
                subject = subject_regex.sub("", line).strip()
                data.append((subject,is_spam))


random.seed(0)

train = data[0:int(0.75*len(data))-1]
test = data[int(0.75*len(data)): len(data)-1]

classifier =  NBClassifier()
classifier.train(train)

classified = [(subject, is_spam, classifier.classify(subject)) for subject,is_spam in test]

counts = Counter((is_spam, spam_probability > 0.5) for _, is_spam, spam_probability in classified)
print(counts)








