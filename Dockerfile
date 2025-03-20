FROM python:3.9

WORKDIR /app
COPY . /app

CMD ["python","feature_selection_revised.py","h20.py","main.py","Step_1.py"]
