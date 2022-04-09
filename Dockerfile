FROM python:3.7

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

# RUN addgroup --system app && adduser --system app && adduser app app
COPY Pipfile Pipfile.lock ./

RUN pip install pipenv==2021.5.29 && pipenv install --system

COPY . ./

# RUN chown -R app:app .
# USER app