## Song Recommendation Systems: A Qualitative Study of Different NLP Approaches based on Lyrics Semantic Similarity

This is the official repo for the NLP course 2023/2024 - final project at University of Padua.

This study aims at evaluating from a qualitative perspective three song recommendation systems designed by us, that use song lyrics as unique source of knowledge. The first system uses the sentiment scores, topic modeling (LDA) and TF-IDF vector to build a unified embedding vector for each lyrics, and then compute cosine similarities among them to sort out the most similar lyrics (embeddings). The other two systems have the same goal, but they differ in the creation of the embeddings by using Fast Text and Universal Sentence Encoder models. More details can be found in the original paper `report.pdf`.

## Getting started

### Environment setup
In order to use our systems, you need to create a conda environment containing all the necessary packages (make sure you have conda up and running!). To simplify your life, we created the file `environment.yml`, that can be used as follow to create the environment for you:

```bash
conda env create -f environment.yml
```

Alternatively, you could create it by yourself (python version is 3.12) and install all the required packages present in `requirements.txt` via pip as follows:

```bash
conda create --name nlp python=3.12
pip install -r requirements.txt
```

### Fast Text and USE models
In order to execute Fast Text and USE models, you need to download their corresponding files locally. You can do it by lunching the scripts:

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

The links to the 5 evaluation forms are available at the following links (submission has been closed):

- Form [Query 1]("https://forms.gle/4yNYVwWRPDa6nHKc8")
- Form [Query 2]("https://forms.gle/ndfx2qw2ZSvts49aA")
- Form [Query 3]("https://forms.gle/4nV9Wb9j6LFoMDjLA")
- Form [Query 4]("https://forms.gle/rF6arS1dTyCiBR3U9")
- Form [Query 5]("https://forms.gle/X3uFKrDRxk6zVG8M8")

The results can be viewed in [this shared Google sheet]("https://docs.google.com/spreadsheets/d/1Bfy9vEGwOHaGMFiaMv3-XBb-j32NRnx5C8EMKMRV0-o/edit?usp=sharing").


