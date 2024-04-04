import pandas as pd
    

if __name__ == "__main__":
    columns = {
        "name": ["A", "B", "C"],
        "age": [11, 13, 17]
    }

    dataFrame = pd.read_csv("output/output_tune_cora_naive_collect.csv",
                            usecols=["exe_time"])
    print(F"dataFrame['exe_time']: {list(dataFrame['exe_time'])}")
    
