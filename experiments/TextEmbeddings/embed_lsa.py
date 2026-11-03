"""LSA: text embedding of the catalogue movies (classical baseline).

TF-IDF (unigrams + bigrams) + truncated SVD, fitted on training movies only; the
number of components is chosen on a validation subset of the training movies.
Replaces Doc2Vec, whose gensim package does not install on Python 3.14.
Outputs results/TextEmbeddings/text_lsa_<input>.
Usage: python -m experiments.TextEmbeddings.embed_lsa --input {overview,meta_core,meta_rich,all}
"""

from experiments.TextEmbeddings.common import main

if __name__ == "__main__":
    main("lsa", __doc__)
