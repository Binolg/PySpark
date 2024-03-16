from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, HasInputCols, Param, Params
import pyspark.sql.functions as F
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.window import Window

# https://stackoverflow.com/questions/32331848/create-a-custom-transformer-in-pyspark-ml
class OneHotLoanType(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
    A custom Transformer that converts 'Type_of_loan' into one-hot encoded columns.
    """
    def __init__(self, inputCol="Type_of_loan", outputCol="loan_type_vector", loanTypes=[]):
        super(OneHotLoanType, self).__init__()
        self.setInputCol(inputCol)
        self.setOutputCol(outputCol)
        self.loanTypes = ["credit-builder loan", "not specified", "mortgage loan", "auto loan", "student loan",
                          "home equity loan", "personal loan", "payday loan", "debt consolidation loan"]
    
    def setInputCol(self, value):
        return self._set(inputCol=value)
    
    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def _transform(self, df):
        inputCol = self.getInputCol()
        clean_input_field = F.regexp_replace(F.lower(F.col(inputCol)), " and ", " ")
        
        for loan_type in self.loanTypes:
            df = df.withColumn(loan_type, F.array_contains(F.split(clean_input_field, ", "), loan_type).cast("int"))
            df = df.withColumn(loan_type, F.coalesce(F.col(loan_type), F.lit(0)))
            df = df.withColumnRenamed(loan_type, loan_type.replace(" ", "_").replace("-", "_"))

        return df

class FillMissingCategorical(Transformer, HasInputCols, Params, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCol=None, fillValue="NA"):
        super(FillMissingCategorical, self).__init__()
        self.inputCol = Param(self, "inputCol", "")
        self.fillValue = Param(self, "fillValue", "")
        self._setDefault(inputCol=inputCol, fillValue=fillValue)
        
    def setInputCol(self, value):
        return self._set(inputCol=value)
    
    def setFillValue(self, value):
        return self._set(fillValue=value)
    
    def _transform(self, df):
        fill_value = self.getOrDefault(self.fillValue)
        target_col = self.getOrDefault(self.inputCol)
        
        df = df.withColumn(target_col, F.when((F.col(target_col).isNull()) | (F.col(target_col) == ""), F.lit(fill_value)).otherwise(F.col(target_col)))

        return df


class ZeroFillImputer(Transformer, HasInputCols, Params, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None):
        super(ZeroFillImputer, self).__init__()
        self.inputCols = Param(self, "inputCols", "")
        self._setDefault(inputCols=inputCols)
        
    def setInputCols(self, value):
        return self._set(inputCols=value)
    
    def _transform(self, df):
        input_cols = self.getOrDefault(self.inputCols)
        for col in input_cols:
            df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
        return df
    

class ImputerWithCustomerData(Transformer, HasInputCols, Params, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None):
        super(ImputerWithCustomerData, self).__init__()
        self.inputCols = Param(self, "inputCols", "")
        self._setDefault(inputCols=inputCols)
        
    def setInputCols(self, value):
        return self._set(inputCols=value)
    
    def _transform(self, df):
        input_cols = self.getOrDefault(self.inputCols)
        
        for col in input_cols:
            # Calculate the global mean for the column
            global_mean = df.select(F.mean(col).alias(f'{col}_global_mean')).collect()[0][f'{col}_global_mean']
            
            # Calculate the mean of values grouped by Customer_ID
            windowSpec  = Window.partitionBy("Customer_ID")
            mean_values = df.withColumn(f"{col}_customer_mean", F.mean(df[col]).over(windowSpec))
            
            # Replace the missing values by the customer mean, if possible, otherwise use the global mean
            df = mean_values.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_customer_mean"), F.lit(global_mean)))
            
            # Drop the intermediate mean column
            df = df.drop(f"{col}_customer_mean")

            
        return df

class EnsureIntegerType(Transformer, HasInputCols, Params, DefaultParamsReadable, DefaultParamsWritable):
    def __init__(self, inputCols=None):
        super(EnsureIntegerType, self).__init__()
        self.inputCols = Param(self, "inputCols", "")
        self._setDefault(inputCols=inputCols)
        
    def setInputCols(self, value):
        return self._set(inputCols=value)
    
    def _transform(self, df):
        input_cols = self.getOrDefault(self.inputCols)
        for col in input_cols:
            df = df.withColumn(col, F.col(col).cast('int'))
        return df

