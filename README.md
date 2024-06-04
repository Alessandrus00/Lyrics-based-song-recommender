## Song Recommendation Systems: A Qualitative Study of Different NLP Approaches based on Lyrics Semantic Similarity

This is the official repo for the NLP course 2023/2024 - final project at University of Padua.

This study aims at evaluating from a qualitative perspective three song recommendation systems designed by us, that use song lyrics as unique source of knowledge. The first system uses the sentiment scores, topic modeling (LDA) and TF-IDF vector to build a unified embedding vector for each lyrics, and then compute cosine similarities among them to sort out the most similar lyrics (embeddings). The other two systems have the same goal, but they differ in the creation of the embeddings by using Fast Text and Universal Sentence Encoder models. More details can be found in the original paper `report.pdf`.

## Repo structure: important files

- Training data: `data/lyrics_proc_train.csv`
- Test data (queries): `data/lyrics_proc_test.csv`
- TF-IDF train embeddings: `tfidf_embeddings.npy.zip`
- fastText train embeddings: `fasttext_embeddings.npy`
- USE train embeddings: `use_embeddings.npy`
- fastText model: `models/fasttext/cc.en.300.bin` (after download)
- USE model: `models/use` (after download)
- LDA model and TF-IDF vectors: `models/others`
- Notebook for dataset preprocessing: `notebooks/dataset_preprocessing.ipynb`
- Python scripts: `scripts`
- Test songs (queries) with best and bad recommendations used in the evaluation forms: `test`

## Getting started

### Environment setup
In order to use our systems, you need to create a conda environment containing all the necessary packages (make sure you have conda up and running!). To facilitate the process, hit the following bash commands in the root folder of the project:

```bash
conda create --name nlp python=3.12
conda activate nlp
pip install -r requirements.txt
```

### Fast Text and USE models
In order to execute Fast Text and USE models, you need to download their corresponding files locally. You can do it by launching the scripts:

For the Fast text model (cc.en.300.bin)
```bash
bash models/download_fasttext.sh
```

For the Universal Sentence Encoder model (transformer-based variant v2)
```bash
bash models/download_use.sh
```

### Get your recommendations!
Great, now you can run the python file `scripts/evaluate.py` and provide it with the appropriate arguments (basically method and number of recommendations to retrieve). 

Here it is an example that retrieve the top-10 similar songs to the song starting with "She hit me like a blinding light and I was born" using the **tfidf**-based method.

The provided recommendations contain just the song IDs, their lyrics can be found in `data/lyrics_proc_train.csv`. 

**Note:** In general you should provide the full lyrics, but if the song is either in the training set `data/lyrics_proc_train.csv` or test set `data/lyrics_proc_test.csv`, a snippet of its lyrics is sufficient (the fulll lyrics will be retrieved automatically).

```bash
python scripts/evaluate.py \
--eval topn \
--n 10 \
--method tfidf \
--lyrics "She hit me like a blinding light and I was born"
```

### Jupiter notebooks
To see more in details what has been done code-wise, and follow more clearly all the steps that make this project work, we created some jupyter notebooks under `notebooks`. Enjoy!

### Evaluation forms and results

The results can be viewed in [this shared Google sheet](https://docs.google.com/spreadsheets/d/1Bfy9vEGwOHaGMFiaMv3-XBb-j32NRnx5C8EMKMRV0-o/edit?usp=sharing).


