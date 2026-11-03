"""bge-large-en-v1.5: text embedding of the catalogue movies.

Pretrained bge-large-en-v1.5 sentence embeddings (nothing fitted on our data).
Outputs results/TextEmbeddings/text_bge_large_<input>.
Usage: python -m experiments.TextEmbeddings.embed_bge_large --input {overview,meta_core,meta_rich,all}
"""

from experiments.TextEmbeddings.common import main

if __name__ == "__main__":
    main("bge_large", __doc__)
