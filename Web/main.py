import streamlit as st
import csv
import pandas as pd
from kafka import KafkaProducer
from json import dumps
from time import sleep
from random import seed
from random import randint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import squarify
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql.functions import split,col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql.functions import isnan, when, count, col

# Start Spark session
scala_version = '2.12'  # your scala version
spark_version = '3.0.1' # your spark version
packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:2.8.0' #your kafka version
]
spark = SparkSession.builder.master("local").appName("kafka-example").config("spark.jars.packages", ",".join(packages)).getOrCreate()


topic_name = 'RandomNumber'
topic_send = 'SendChart'
topic_result = 'SendResult'
kafka_server = 'localhost:29092'

producer = KafkaProducer(bootstrap_servers=kafka_server,value_serializer = lambda x:dumps(x).encode('utf-8'))

st.title('CSV to Kafka Demo')

uploaded_file = st.file_uploader('Choose a CSV file', type=['csv'])
if uploaded_file is not None:
    # bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)
    
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_csv(uploaded_file)
    with open(uploaded_file.name, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            print(row)
            producer.send(topic_name, row)
    st.write(dataframe)

st.header('Some plot')
df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic_name) \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load() 

df = df.selectExpr("CAST(value AS STRING) as json")
df = df.selectExpr("from_json(json, 'gender STRING, age STRING, hypertension STRING, heart_disease STRING, ever_married STRING, work_type STRING, Residence_type STRING, avg_glucose_level STRING, bmi STRING, smoking_status STRING, stroke STRING') as data")

df = df.select(
    col("data.gender"),
    col("data.age").cast(DoubleType()).alias("age"),
    col("data.hypertension").cast(IntegerType()).alias("hypertension"),
    col("data.heart_disease").cast(IntegerType()).alias("heart_disease"),
    col("data.ever_married"),
    col("data.work_type"),
    col("data.Residence_type"),
    col("data.avg_glucose_level").cast(DoubleType()).alias("avg_glucose_level"),
    col("data.bmi").cast(DoubleType()).alias("bmi"),
    col("data.smoking_status"),
    col("data.stroke").cast(IntegerType()).alias("stroke")
)

df_pd = df.toPandas()

df_pd.loc[(df_pd.stroke == 1), 'stroke']='Stroke'
df_pd.loc[(df_pd.stroke == 0), 'stroke']='No stroke'

df_pd.loc[(df_pd.heart_disease == 1), 'heart_disease']='Heart disease'
df_pd.loc[(df_pd.heart_disease == 0), 'heart_disease']='No heart disease'

df_pd.loc[(df_pd.hypertension == 1), 'hypertension']='Hypertension'
df_pd.loc[(df_pd.hypertension == 0), 'hypertension']='No hypertension'

fig, axes = plt.subplots(2, 4, figsize=(20,10))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
sns.set_theme()

sns.countplot(data=df_pd, x='stroke', ax=axes[0,0])

sns.countplot(data=df_pd, x='hypertension', order=['Hypertension', 'No hypertension'], ax=axes[0,1])

sns.countplot(data=df_pd, x='heart_disease', ax=axes[0,2])

sns.countplot(data=df_pd, x='gender', ax=axes[0,3])

sns.countplot(data=df_pd, x='ever_married', ax=axes[1,0])

sns.countplot(data=df_pd, x='work_type', ax=axes[1,1])
axes[1,1].tick_params(axis='x', rotation=30)

sns.countplot(data=df_pd, x='Residence_type', ax=axes[1,2])

sns.countplot(data=df_pd, x='smoking_status', ax=axes[1,3])
axes[1,3].tick_params(axis='x', rotation=30)
st.pyplot(fig)


fig, axes = plt.subplots(3, 1, figsize=(15,10))
plt.subplots_adjust(hspace=0.4)
sns.set_theme()

axes[0].set_title('Age distribution')
sns.histplot(df_pd['age'], bins=40, kde=True, alpha=0.7, ax=axes[0])

axes[1].set_title('Glucose level distribution')
sns.histplot(df_pd['avg_glucose_level'], bins=40, kde=True, alpha=0.7, ax=axes[1])

axes[2].set_title('BMI distribution')
sns.histplot(df_pd['bmi'], bins=40, kde=True, alpha=0.7, ax=axes[2])

st.pyplot(fig)


fig, axes = plt.subplots(1, 3, figsize=(30,7))

sns.boxplot(data=df_pd, x='stroke', y='age', ax=axes[0])

sns.boxplot(data=df_pd, x='stroke', y='avg_glucose_level', ax=axes[1])

sns.boxplot(data=df_pd, x='stroke', y='bmi', ax=axes[2])

st.pyplot(fig)

fig = sns.displot(df_pd, x='age', hue='stroke', col='stroke', stat='density', common_norm=False)
st.pyplot(fig)


fig = sns.displot(df_pd, x='avg_glucose_level', hue='stroke', col='stroke', stat='density', common_norm=False)
st.pyplot(fig)

df = df.withColumn(
    'high_risk',
    when((col('age') >= 40) & (col("avg_glucose_level") <= 125), 1).otherwise(0))
fig, axes = plt.subplots(1, 2, figsize=(30,7))

axes[0].set_title('Heart disease and strokes', fontsize=20)
ax0 = sns.countplot(data=df.toPandas().loc[df_pd['stroke'] == 'Stroke'], x='high_risk', ax=axes[0])

axes[1].set_title('Heart disease and no strokes', fontsize=20)
ax1 = sns.countplot(data=df.toPandas().loc[df_pd['stroke'] == 'No stroke'], x='high_risk', ax=axes[1])
st.pyplot(fig)

fig, axes = plt.subplots(1, 2, figsize=(30,7))

axes[0].set_title('Gender and strokes', fontsize=20)
sns.countplot(data=df_pd.loc[df_pd['stroke'] == 'Stroke'], x='gender', ax=axes[0])

axes[1].set_title('Gender and no strokes', fontsize=20)
sns.countplot(data=df_pd.loc[df_pd['stroke'] == 'No stroke'], x='gender', ax=axes[1])
st.pyplot(fig)

fig, axes = plt.subplots(1, 2, figsize=(30,7))

axes[0].set_title('Hypertension and strokes', fontsize=20)
sns.countplot(data=df_pd.loc[df_pd['stroke'] == 'Stroke'], x='hypertension', ax=axes[0])

axes[1].set_title('Hypertension and no strokes', fontsize=20)
sns.countplot(data=df_pd.loc[df_pd['stroke'] == 'No stroke'], x='hypertension', ax=axes[1])
st.pyplot(fig)

fig, axes = plt.subplots(1, 2, figsize=(30,7))

axes[0].set_title('Heart disease and strokes', fontsize=20)
ax0 = sns.countplot(data=df_pd.loc[df_pd['stroke'] == 'Stroke'], x='heart_disease', order=['No heart disease', 'Heart disease'], ax=axes[0])

axes[1].set_title('Heart disease and no strokes', fontsize=20)
ax1 = sns.countplot(data=df_pd.loc[df_pd['stroke'] == 'No stroke'], x='heart_disease', ax=axes[1])
st.pyplot(fig)

df = spark.read \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic_result) \
    .option("startingOffsets", "earliest") \
    .option("endingOffsets", "latest") \
    .load() 


df = df.selectExpr("CAST(value AS STRING) as json")
df = df.selectExpr("from_json(json, 'Models STRING, Accuracy STRING, Precision STRING, ROC_Score STRING') as data")
df = df.select(
    col("data.Models"),
    col("data.Accuracy").cast(DoubleType()).alias("Accuracy"),
    col("data.Precision").cast(DoubleType()).alias("Precision"),
    col("data.ROC_Score").cast(DoubleType()).alias("ROC_Score"),
)
st.header('Result form server')
st.write(df)
df.show()