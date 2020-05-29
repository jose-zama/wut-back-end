FROM python:3 AS base
WORKDIR /usr/src/app

# Download Dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz

COPY post-install.py ./
RUN python post-install.py

# Run server to expose API
CMD ./start.sh

### DEV ###
FROM base AS dev
COPY requirements-dev.txt ./requirements-dev.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

ENV FLASK_ENV="development"

### PROD ###
FROM base AS prod

# Copy project
# We should have a dist folder and just copy it
COPY src/ ./src/
COPY Output/ ./Output/
COPY Datasets/ ./Datasets/
COPY *.py ./
COPY start.sh ./start.sh
