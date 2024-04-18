import pandas as pd


def fun1():
    print("fun1 call")

def fun2():
    print("fun2 call")

if __name__ == "__main__":
    # funs = {
    #     "fun1": fun1,
    #     "fun2": fun2
    # }

    # for key, val in funs.items():
    #     print("Go to {key}")
    #     val()
    # df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    df = pd.DataFrame()
    df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
    print(F"df:\n{df}")
    print(F"df2:\n{df2}")
    for i in range(df2.shape[0]):
        df = pd.concat([df, df2.iloc[i:i+1]], ignore_index=False)
        # df = pd.concat([df, df2.iloc[i:i+1]], ignore_index=True)

    print(F"df:\n{df}")