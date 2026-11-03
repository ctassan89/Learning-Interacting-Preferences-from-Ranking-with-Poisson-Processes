"""all-MiniLM-L6-v2: text embedding of the catalogue movies.

Pretrained all-MiniLM-L6-v2 sentence embeddings (nothing fitted on our data).
Outputs results/TextEmbeddings/text_minilm_<input>.
Usage: python -m experiments.TextEmbeddings.embed_minilm --input {overview,meta_core,meta_rich,all}
"""

from experiments.TextEmbeddings.common import main

if __name__ == "__main__":
    main("minilm", __doc__)
