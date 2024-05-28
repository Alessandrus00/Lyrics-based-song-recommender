# Pute here your Fast Text model (cc.en.300.bin file)

Can be downloaded by uncommenting the following line in fasttext.ipynb:
```python
fasttext.util.download_model(
    'en', 
    if_exists='ignore', 
    model_dir=./models/
    )
```

Or by using the following bash command from the root directory:
```bash
wget -P ./models/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip ./models/cc.en.300.bin.gz
```