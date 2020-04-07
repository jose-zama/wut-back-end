FROM python:3

WORKDIR /usr/src/app

# Download Dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

COPY post-install.py ./
RUN python post-install.py

# Copy project
# We should have a dist folder and just copy it
COPY Adapter/ ./Adapter/
COPY App/ ./App/
COPY Conf ./Conf/
COPY Usecase ./Usecase/
COPY PyTorchModelLSTM/src/ ./PyTorchModelLSTM/src/
COPY PyTorchModelLSTM/Output/ ./PyTorchModelLSTM/Output/
COPY PyTorchModelLSTM/Datasets/ ./PyTorchModelLSTM/Datasets/
COPY PyTorchModelLSTM/*.py PyTorchModelLSTM/
COPY __init__.py ./

# Run server
WORKDIR /usr/src/app/App
CMD ./start_prod.sh