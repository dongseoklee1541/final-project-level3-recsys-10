# pull base image
FROM python:3.8.5-slim-buster

COPY start.sh /beerrecsys/start.sh
COPY /backend /beerrecsys/backend
COPY /frontend /beerrecsys/frontend
COPY requirements.txt /beerrecsys/requirements.txt

RUN pip install --upgrade pip \ 
&& pip install -r /beerrecsys/requirements.txt

WORKDIR /beerrecsys


EXPOSE 8501
CMD ["sh", "start.sh"]
# CMD ["python", "-m", "backend.app"]
