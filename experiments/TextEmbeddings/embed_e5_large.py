"""e5-large-v2: text embedding of the catalogue movies.

Pretrained e5-large-v2 sentence embeddings (nothing fitted on our data).
Outputs results/TextEmbeddings/text_e5_large_<input>.
Usage: python -m experiments.TextEmbeddings.embed_e5_large --input {overview,meta_core,meta_rich,all}
"""

from experiments.TextEmbeddings.common import main

if __name__ == "__main__":
    main("e5_large", __doc__)
