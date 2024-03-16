import io
import streamlit as st

from pyspark.ml import PipelineModel
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark import SparkConf
from pyspark.sql import SparkSession

from app_aux import OneHotLoanType, FillMissingCategorical, ZeroFillImputer, ImputerWithCustomerData, EnsureIntegerType


# Initialize Spark Session
@st.cache_resource
def get_spark_session():
    proj_conf = SparkConf().setAppName("projBigData_deploy").set("spark.driver.memory", "16g").set("spark.jars.packages", "org.apache.spark:spark-avro_2.12:3.5.0")
    return SparkSession.builder.config(conf=proj_conf).getOrCreate()

spark = get_spark_session()

# Load model and pipeline
@st.cache_resource
def load_model(model_path):
    print(f"Loading Model from {model_path}")
    return MultilayerPerceptronClassificationModel.load(model_path)

@st.cache_resource
def load_pipeline(pipeline_path):
    print(f"Loading Pipeline from {pipeline_path}...")
    return PipelineModel.load(pipeline_path)

def df_to_str(df):
    data = df.collect()
    output = io.StringIO()
    output.write(" | ".join(df.columns) + "\n")
    output.write("-" * 50 + "\n")
    
    for row in data:
        row_data = " | ".join(str(cell) for cell in row)
        output.write(row_data + "\n")
    
    return output.getvalue()
    

pipeline_path = "./pipeline_model"
model_path = "./MultiLayerPercep_CV_PG"
pipeline = load_pipeline(pipeline_path)
model = load_model(model_path)


### OPTIONS:
months = ("January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December")
occupations = ("Scientist", "Media_Manager", "Musican", "Lawyer", "Teacher", "Developer", "Writer",
               "Architect", "Mechanic", "Entrepreneur", "Journalist", "Doctor", "Engineer", "Accountant",
               "Manager", "NA")
loan_types = ("Debt Consolidation Loan", "Payday Loan", "Not Specified", "Mortgage Loan",
              "Credit-Builder Loan", "Personal Loan", "Home Equity Loan", "Student Loan",
              "Auto Loan")
credit_mix_types = ("Bad", "Standard", "Good", "NA")
payment_behaviours = ("Low_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Small_value_payments",
                      "Low_spent_Large_value_payments", "Low_spent_Medium_value_payments", "High_spent_Large_value_payments",
                      "NA")


st.title('BigData - Credit Score Classifier')

month_op = st.selectbox("Month (Month)", months)
age_op = st.number_input("Age (Age)", min_value=0, format="%d")
occupation_op = st.selectbox("Occupation (Occupation)", occupations)
annual_income_op = st.number_input("Annual Income (Annual_Income)", min_value=0.0)
monthly_inhand_salary_op = st.number_input("Monthly Inhand Salary (Monthly_Inhand_Salary)", min_value=0.0)
num_bank_accounts_op = st.number_input("Number of bank accounts (Num_Bank_Accounts)", min_value=0, format="%d")
num_credit_cards_op = st.number_input("Number of credit cards (Num_Credit_Card)", min_value=0, format="%d")
interest_rate_op = st.number_input("Interest rate (Interest_Rate)", min_value=0.0)
type_of_loan_op = st.multiselect("Types of Loans: (Type_of_Loan)", loan_types)
delay_from_due_date_op = st.number_input("Delay from due date (Delay_from_due_date)", min_value=0, format="%d")
num_delayed_payments_op = st.number_input("Previous delayed payments (num_delayed_payments)", min_value=0, format="%d")
changed_credit_limit_op = st.number_input("Changed credit limit (Changed_Credit_Limit)", min_value=0.0)
num_credit_inquiries_op = st.number_input("Previous credit inquiries (num_credit_inquiries)", min_value=0, format="%d")
credit_mix_op = st.selectbox("Previous credit mix: (Credit_mix)", credit_mix_types)
outstanding_debt_op = st.number_input("Outstanding debpt (outstanding_debt)", min_value=0.0)
credit_utilization_ratio_op = st.number_input("Credit utilization ratio (credit_utilization_ratio)", min_value=0.0)
credit_history_age_op = st.number_input("Credit history (months) (credit_history_age)", min_value=0, format="%d")
payment_of_min_amount_op = st.selectbox("Payment of minimun amount (payment_of_min_amount)", ("Yes", "No", "NA"))
total_emi_per_month_op = st.number_input("Total EMI per month (total_emi_per_month)", min_value=0.0)
amount_invested_monthly_op = st.number_input("Amount invested monthly (amount_invested_monthly)", min_value=0.0)
payment_behaviour_op = st.selectbox("Payment behaviour (payment_behaviour)", payment_behaviours)
monthly_balance_op = st.number_input("Monthly balance (monthly_balance)", min_value=0.0)


columns = ['Customer_ID',
           'Month',
           'Age',
           'Occupation',
           'Annual_Income',
           'Monthly_Inhand_Salary',
           'Num_Bank_Accounts',
           'Num_Credit_Card',
           'Interest_Rate',
           'Num_of_Loan',
           'Type_of_Loan',
           'Delay_from_due_date',
           'Num_of_Delayed_Payment',
           'Changed_Credit_Limit',
           'Num_Credit_Inquiries',
           'Credit_Mix',
           'Outstanding_Debt',
           'Credit_Utilization_Ratio',
           'Credit_History_Age',
           'Payment_of_Min_Amount',
           'Total_EMI_per_month',
           'Amount_invested_monthly',
           'Payment_Behaviour',
           'Monthly_Balance',
           'Credit_Score']


#####################################
## ACTION

if st.button("Classify!"):
    try:
        # Construct DataFrame for prediction
        new_data = [(1, # Customer ID - to ignore
                     month_op,
                     age_op,
                     occupation_op,
                     annual_income_op,
                     monthly_inhand_salary_op,
                     num_bank_accounts_op,
                     num_credit_cards_op,
                     interest_rate_op,
                     len(type_of_loan_op),
                     ", ".join(type_of_loan_op),
                     delay_from_due_date_op,
                     num_delayed_payments_op,
                     changed_credit_limit_op,
                     num_credit_inquiries_op,
                     credit_mix_op,
                     outstanding_debt_op,
                     credit_utilization_ratio_op,
                     credit_history_age_op,
                     payment_of_min_amount_op,
                     total_emi_per_month_op,
                     amount_invested_monthly_op,
                     payment_behaviour_op,
                     monthly_balance_op,
                     "---" # credit score (target) to be ignored. Just needed for the pipeline
                     )]
        
        new_df = spark.createDataFrame(new_data, columns)
        with open("log.txt", "a") as file:
            file.write(f"Raw Input:\n{df_to_str(new_df)}")

        transformed_new_df = pipeline.transform(new_df)
        with open("log.txt", "a") as file:
            file.write(f"Transformed Input:\n{df_to_str(transformed_new_df)}")
        predicted_new_df = model.transform(transformed_new_df)
        
        prediction = int(predicted_new_df.select("prediction").first()[0])

        prediction_dic = {0: "(0) Poor :(", 1: "(1) Standard :|", 2: "(2) Good :)", 3: "????"}
        prediction_str = prediction_dic.get(prediction, f'Oops... Something went wrong. Class {prediction} unknown')
        st.success(f"Classification Credit Score: {prediction_str}")
    except Exception as e:
        st.error(f"An error occurred: {e}")


## RUN
# streamlit run app.py