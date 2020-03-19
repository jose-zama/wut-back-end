# Installation

Clone the repository and then:

With python installed:
```
# pip install -r requirements.txt
# pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz
# python  
  >>> import nltk
  >>> nltk.download('stopwords')
  >>> exit()
# cd App && ./start.sh 
```

OR With Docker:
```
# docker build -t classifier-image .
# docker run -d --rm -p 5000:5000 -v "$PWD":/usr/src/app/ --name classifier classifier-image
```

# Play
```
curl --location --request POST 'localhost:5000/classification/movements' \
--header 'Content-Type: text/plain' \
--data-raw 'Details
walmart
uber'
```


