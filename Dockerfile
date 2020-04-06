FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
COPY Adapter ./
COPY App ./
COPY Conf ./
COPY PyTorchModelLSTM/src ./
COPY PyTorchModelLSTM/Output ./
COPY PyTorchModelLSTM/Datasets ./

RUN pip install --no-cache-dir -r requirements.txt

RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

COPY post-install.py ./
RUN python post-install.py

WORKDIR /usr/src/app/App
CMD ./start_prod.sh