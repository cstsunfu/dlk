# Data download

Download [Conll2003](https://data.deepai.org/conll2003.zip) and unzip to datadir
 
# Convert to json

```
python ./bio2json.py 
```

# Do prepro use the prepro.hjson config

# Do train use the main.hjson config

# Convert the result to bio with json2bio.py

# Eval using conlleval.pl script

 This is because the dlkit evaluation is not very same as conlleval script result
