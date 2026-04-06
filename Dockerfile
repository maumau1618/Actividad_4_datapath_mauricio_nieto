FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

# set the working directory
WORKDIR /usr/src/app

# install dependencies
COPY ./requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include the src directory
ENV PYTHONPATH=/usr/src/app/src

# copy src code
COPY ./src ./src

EXPOSE 8080

# start the server FastAPI
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]