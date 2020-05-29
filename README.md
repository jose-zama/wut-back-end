# Development

Clone the repository and then:

## With Docker:
```
# docker-compose up
```

## In a traditional way: 
With python installed (we recommend to use a venv enviroment):
```
# pip install -r requirements-dev.txt
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
# python  
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> exit()
# ./start.sh 
```

# Play
```
curl --location --request POST 'localhost:5000/classification/movements' \
--header 'Content-Type: text/plain' \
--data-raw 'walmart
uber'
```

# Test

To run classifier model tests:
```
python -m unittest
```

# Production

`Dockerfile` copy the project instead, but still a volume has to be used to write `.vector_cache`.
```
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

