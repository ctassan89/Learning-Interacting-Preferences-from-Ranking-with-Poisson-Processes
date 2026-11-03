"""all-mpnet-base-v2: text embedding of the catalogue movies.

Pretrained all-mpnet-base-v2 sentence embeddings (nothing fitted on our data).
Outputs results/TextEmbeddings/text_mpnet_<input>.
Usage: python -m experiments.TextEmbeddings.embed_mpnet --input {overview,meta_core,meta_rich,all}
"""

from experiments.TextEmbeddings.common import main

if __name__ == "__main__":
    main("mpnet", __doc__)
