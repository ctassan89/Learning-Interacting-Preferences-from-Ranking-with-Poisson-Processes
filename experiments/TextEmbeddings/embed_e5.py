"""e5-base-v2: text embedding of the catalogue movies.

Pretrained e5-base-v2 sentence embeddings (nothing fitted on our data).
Outputs results/TextEmbeddings/text_e5_<input>.
Usage: python -m experiments.TextEmbeddings.embed_e5 --input {overview,meta_core,meta_rich,all}
"""

from experiments.TextEmbeddings.common import main

if __name__ == "__main__":
    main("e5", __doc__)
