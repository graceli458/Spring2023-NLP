import sys
from collections import defaultdict
import math
import random
import os
import os.path
import itertools

"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    #append START and STOP sequences to the string
    if(n<2):
        sequence.insert(0, "START")
        sequence.append("STOP")
    else:
        #print("hi")
        for i in range(n-1):
            sequence.insert(0, "START")
        sequence.append("STOP")
    
    #create tuples based on the length of n
    temp = []
    for i in range(len(sequence)-(n-1)):
        temp.append(tuple(sequence[i:i+n]))

    return temp


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        self.sentenceTotal = 0
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        # Get size of lexicon
        self.total = 0
        for key,value in self.unigramcounts.items():
            self.total += value



    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int) 
        self.trigramcounts = defaultdict(int) 


        for sentence in corpus:
            
            #check count of sentence is equal to total number of START, total number of STOP
            self.sentenceTotal += 1
            
            unigram = get_ngrams(sentence, 1)
            for word in unigram:
                self.unigramcounts[word]+=1
            
            bigram = get_ngrams(sentence, 2)
            for word in bigram:
                self.bigramcounts[word]+=1
            
            trigram = get_ngrams(sentence, 3)
            for word in trigram:
                self.trigramcounts[word]+=1
          
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        #edge case: START START something -> just want raw bigram of START something
        #edge case: count bigram = 0 -> just want to get raw unigram of word !! verify

        numerator = 0
        denominator = 0

        #case when trigram begins ('START', 'START', word) -> take the bigram of ('START', word)
        if(trigram[0] == 'START' and trigram[1] == 'START'):
            return self.raw_bigram_probability(tuple([trigram[1], trigram[2]]))
        
        
        numerator = self.trigramcounts[trigram]
        denominator = self.bigramcounts[tuple([trigram[0], trigram[1]])]
        
        #case if unigram doesn't occur to in training data
        if(numerator == 0):
            return 1 / self.total

        #if denominator is zero, that menas the bigram doesn't occur in training data
        if(denominator == 0):
            denominator = self.unigramcounts(tuple(trigram[0]))

        return numerator/denominator

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        
        numerator = self.bigramcounts[bigram]
        denominator = self.unigramcounts[tuple([bigram[0],])]

        #the occurance of the bigram divided by total number of sentences
        if(bigram[0]=="START"):
            denominator = self.sentenceTotal
            return numerator / denominator

        if(denominator == 0):
            return 1 / self.total

        return numerator/denominator
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
         
        numerator = self.unigramcounts[unigram]

        return numerator / self.total

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        return lambda1 * self.raw_trigram_probability(trigram) + lambda2 * self.raw_bigram_probability(tuple([trigram[1], trigram[2]])) + lambda3 * self.raw_unigram_probability(tuple([trigram[2]]))
      
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        total = 0
        for trigram in get_ngrams(sentence, 3):
            smoothed = self.smoothed_trigram_probability(trigram)
            if(smoothed>0):
                total = total + math.log2(smoothed)
               
        return float(total)

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        total = 0
        lengthOfLexicon = 0 
        
        for sentence in corpus:
            total += self.sentence_logprob(sentence)
            lengthOfLexicon += len(sentence)
            
        return float(2**(-(total / lengthOfLexicon)))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        #high
        model1 = TrigramModel(training_file1)
        #low
        model2 = TrigramModel(training_file2)

        #high: testdir1
        #low: testdir

        total = 0
        correct = 0    
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            if(pp <= pp_2):
                correct += 1

            total += 1

        for f in os.listdir(testdir2):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            
            if(pp_2 <= pp):
                correct += 1

            total += 1
        return correct / total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # print(get_ngrams(["natural", "language", "processing"], 1))

    # print(model.bigramcounts[('START','the')])
    # print(model.trigramcounts[('START','START', 'the')])
    # print(model.unigramcounts[('START'),])
    # print(model.sentenceTotal)

    # print(model.raw_unigram_probability(('the'),))
    # print(model.raw_bigram_probability(('START','waiting')))
    # print("start start waiting", model.smoothed_trigram_probability(('grace','loves', 'computer')))

    


    #print(model.bigramcounts.items())
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    
    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("perplexity: " +str(pp))
    


    # Essay scoring experiment: 
    acc = essay_scoring_experiment('./hw1_data/ets_toefl_data/train_high.txt', "./hw1_data/ets_toefl_data/train_low.txt", "./hw1_data/ets_toefl_data/test_high", "./hw1_data/ets_toefl_data/test_low")
    print(acc)

