# Installation

Clone the repository and then:

With python installed (we recommend to use a venv enviroment):
```
# pip install -r requirements-dev.txt
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
# python  
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> exit()
# cd App && ./start.sh 
```

OR With Docker:
```
# docker build -f Dockerfile.dev -t classifier-image-dev .
# docker run -d --rm -p 5000:5000 -v "$PWD":/usr/src/app/ --name classifier classifier-image-dev
```

# Play
```
curl --location --request POST 'localhost:5000/classification/movements' \
--header 'Content-Type: text/plain' \
--data-raw 'Details
walmart
uber'
```

# Model development

You need jupyter beforehand.

# Deployment

`Dockerfile` copy the project instead, but still a volume has to be used to write `.vector_cache`
```
docker build -t classifier-image .
docker run -d --rm -p 5000:5000 -v "$PWD":/usr/src/app/App/.vector_cache --name classifier classifier-image
```

