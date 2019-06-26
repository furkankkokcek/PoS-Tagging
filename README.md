# PoS-Tagging
# Introduction
Part-of-speech (PoS) tagging is the task of assigning syntactic categories to words in
a given sentence according to their syntactic roles in that context, such as a noun,
adjective, verb etc. A PoS tagger takes a sentence as input and generates a word/tag
tuples as output:  
  Example An example tagging:  
    İnput: Bunu zaten biliyordum. (I have already known that.)  
    Output: Bunu/Pron zaten/Adv biliyordum/Verb ./Punc  
This application contains implementation of  PoS tagger using Hidden Markov Models
(HMMs).
# Dataset
METU-Treebank is a Turkish dataset that is built from manually collected newspapers,
journal issues, and books. The corpus involves 5659 sentences. You will take %70 of the
data as training (for Task 1) and rest of it as test data (for Task 2 and Task 3). In other
words, the first 3960 lines will be used for training and the last 1699 lines will be used
for testing.

