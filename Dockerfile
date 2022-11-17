FROM python:3.9
WORKDIR /app

COPY . /app
RUN pip install --upgrade pip
#RUN pip install dtale[streamlit]
RUN pip install -r requirements.txt
RUN pip install statsmodels --upgrade
EXPOSE 8501
RUN mkdir ~/.streamlit
RUN cp config.toml ~/.streamlit/config.toml
RUN cp credentials.toml ~/.streamlit/credentials.toml
WORKDIR /app
ENTRYPOINT ["dtale-streamlit", "run"]
CMD ["abihydralitapp.py"]