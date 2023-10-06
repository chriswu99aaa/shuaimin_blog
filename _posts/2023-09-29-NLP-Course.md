## Week 1

### Aims of NLP Course

* Explain the fundamental principles and the major challenges in processing large-scale natural language text.
* Demonstrate how the essential components of NLP systems are built and tuned
* Introduce some principal appplications of NLP
    * information extraction
    * text classification
    * spoken language processing

Necessary steps for *understanding* a piece of data represented by a language

* text mining
* text analytics
* computational linguistics

### Natural Language System

A lanuage processing application requies the **use of knowledge about human language**

* information retrieval: searching document
* document classification: sorting documents into categories
* question answering: short answer for a question
* text summarisation: summarise a set of documents
* sentiment analysis: product reviews, hate cirme detection
* machien translation:
* natural language generation
* authoring and markin tools: check spelling, grammar, style

### Challenges of NLP

* unstructured data
* variability: many ways to express the same thing
* ambibuity: different meanings of words and sentences
    * lexical ambiguity: a word with multiple part of speech tags
    * lexical-semantic ambiguity: a word with different senses
    * syntactic ambiguity: ambiguity coming from possible word groupings.

### Basic Linguistic concepts

* parts of speech (POS)
    * open class words: nouns, verbs, adjectives, adverbs
    * closed class words: pronouns, prepositions
* features/attributes
    * syntactic ambiguity

Syntactic ambiguity

* attachment ambiguity
* coordination ambiguity
* local ambiguity

### Corpus

a large collection of linguistic data

corpus can be 
* unannotated - raw text/speech
* annoated - enhanced with linguistic information

Annotaed (labelled) corpus: is a repository of explicit linguistic information

Annotation types

* grammatical
* semantic
* pragmatics
* combined

we need annotation for training and evaluation.

## Week2

*Question:* Do we know the natural language of the document?

### Dealing with Words: text pre-processing

#### Typical tokenisation steps

1. initial sementation: mostly on white-spaces
2. handinling abbreviations and apostrophes
3. handling hypehnations
4. dealing with other speicial expressions: emails, URLs, emoticons, numbers. 

**Tokenisation:** is the step, in which a sentence is broken down into small chunks of words.

Tokenisation is to know where to split *not* where to combine.

#### Normalisation

* map tokens to normalised forms
    * {walsk, walked, walk}-> walk
* Two principal approaches
    * lemmatisation
    * Stemming
* Case Foldind: convert everything to lowercase
    * **Good** for collecting stas and behaviour of words
    * **Good** for seach engines: users usuallly use lowercase regardless of the correct case of words.
    * **Lose** the ability to identify entity names i.e. organisation names, or people names.

#### Lemmatisation

reduction to "dictionary headword" form (lemma)
    * examples:
        * {am, are, is}-> be
        * {horse, horses, horse's}-> horse
    * How to do lemmatisation
        * Distionary of word forms
            * dictionary look-up might be slow
            * what to do with words not in the dictionary

##### Morphological analysis (词根词缀分析)
        * morphemes: 词根词缀
        * morphemes: stem and affixes (prefix, suffix)

##### Stemming
        * chop "ends of words": remove suffixes, possibly prefixes。 去除前后缀，仅保留词根

Stemming erros:
* under-stemming fails to conflate related forms
    * divide -> divid
    * division -> divis
* over-stemming conflates unrelated forms
    * neutron, neutral 

##### Porter Stemmer

1. get rid of plurals and -ed or -ing suffixes
2. turn terminal y to i when there is another vowel in the stem
3. map double suffixes to single ones
4. deal with suffices, -full, -ness etc
5. take off -ant, -ence, etc
6. tidy up

![image](../pictures/morphems.png)
![image](../pictures/stemming.png)

Byte-Pair Encoding

* token learner: takes a row training corpus and induces a vocabulary
* token segmenter: takes a row test, input sentence and tokenizes it according to that vocabulary.

Repeat
* choose the two symbols that are most frequently adjacent in he training corpus (say 'A', 'B')
* ad a new merged symbol 'AB' to the vocabulary
* replace every adacent 'A' 'B' in the corpus with 'AB'

Split every word into individual character, and based on the frequency of co-occurance we merge these characters.

### N-Gram Language Models and Representations

**Model:** an abstract representation of sth. in computational form.

**Language Model:** a function that assigns a probability over a piece of text so that 'plural' pieces have a larger probability.

**BoW:** Bag of words representation: reduce each 

*Question:* 

* is the meaning lost without order?
* are all words equally important?
* would it work for all language?
* **Bow** is efficient

##### Frequency of Words

**Zipf's Law**: frequency of any word in a given collection is inversely proportional to its rank in the frequency table.

![image](../pictures/zipf.png)

##### Luhn's hypothesis

![image](../pictures/luhn.png)

The words excedding the upper cut-off ar econsidered to be common.

Those below the lower cut-off rare, and therefore not contributig significantly to the content of the article.

#### Removing stop words

Highly frequently occuring words have low distingguishing power for representing documents

* i.e the ,and , of, it
* these words could be filtered.

#### Vector Representation

Vector representation is a way to implement the BoW model.

* Each document is represented |V| dimensional vector

![image](../pictures/vec.png)

#### Term-Document Matrix

![image](../pictures/term_d.png)

#### Inverse Document Frequency

$$idf_t = log_{10}(\frac{N}{df_t})$$

df_t is the number of documents tht contain term t

**tf.idf**: the weight of a term is the product of its tf weight and its idf weight

$$tf.idf_{t,d} = (1+log_{10})\times(log_{10}(\frac{N}{df_t}))$$

![image](../pictures/tf_idf.png)

**Question**: how to get more dense models? Now we only have sparse models.

#### Probabilistic Language Models

A *language model*(LM) meansures the probability of natural langauge utterances, giving higher scores to those that are more common, more grammatical, more "natural"

#### Unigram Language Model

consider the sequence

$$S = w_1 w_2 ... w_n$$

We assume independently and identically distributed (iid) of each other.

![image](../pictures/uni_gram.png)


#### Chain Rule in Probability Theory

$$p(S)=p(the)\times p(cat|the)\times p(in|the, cat) \times p(the | the,cat, in) \times p(hat| the, cat, in the) $$

$$p(w_1, w_2, ... w_L) = p(w_1) \times p(w_2|w_1) ... p(w_L| w_1,w_2,..., w_{L-1}) = \prod_{k=1}^L p(w_k|w_1,w_2,...,w_{k-1})$$

![image](../pictures/uni_bigram.png)

$$p(w_{1:L}) = \prod_{k=1}^L p (w_k|w_{k-N+1:k-1}) = \prod_{k=1}^L p(w_k|w_{k-1})$$

For example

![image](../pictures/bigram_example.png)


Bigram model corresponds to Markov chain. It assumes that a state depends only on its previous state not on the other events that occurred before it.

A **Tri-gram** model takes care of previous two states, and a **N-gram** model takes care of previous N-1 states.

How to estimate the N-gram conditional probabilityusing a training corpus

* using statistical models such as hidden markov model
* using a ML model


##### Count based estimation

![image](../pictures/count_based_est.png)

here we count the count the co-occurance of word w_k with w_k-1 over the occurance of word w_k-1. This proportion can estimate the N-gram probability.


N-gram model is generative. 

#### Evaluation: How good is our model

* extrinsic evaluation of LM
    * put each model in a task: pelling corrector, speech rcognizer
    * run the task, get an accuracy for A and B models
    * ocmpare accuracy for A and B
    * not easy and time consuming

* intrinsic evaluation of LMs
    * the best language model is one that best predicts an unseen test set.
    * gives best probabilityies for test set
    * *But* it doesn't tell us how usefulthe model is
    
#### Smoothing

Laplace smoothing

![image](../pictures/laplace.png)


