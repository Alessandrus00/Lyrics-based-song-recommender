#!/bin/bash

# Execute Python script
python scripts/evaluate.py \
--eval topn \
--n 10 \
--method tfidf \
--lyrics "She hit me like a blinding light and I was born"
