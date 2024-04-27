import pandas as pd
import math


#-------------------------------------Sample equally from all directors to create test and train-------------------------------------
def split_test_train(df, genre, train_split=0.7):
    columns = df.columns.tolist()
    columns_x =[]
    columns_y =[]
    for i in columns:
        if i in genre.columns:
            columns_y.append(i)
        elif i=='title_year':
            columns_y.append(i)
        else:
            columns_x.append(i)

    x_train = pd.DataFrame(columns=columns_x)
    y_train = pd.DataFrame(columns=columns_y)
    x_test = pd.DataFrame(columns=columns_x)
    y_test = pd.DataFrame(columns=columns_y)

    #-------------------------------------Director-wise Addition-------------------------------------
    all_directors = df['director_name'].unique()
    for i in all_directors:
        df_dir = df[df['director_name']==i]
        
        num_train = math.floor(train_split*len(df_dir))
        df_train = df_dir.iloc[:num_train]
        df_test = df_dir.iloc[num_train:]
        
        x_train = pd.concat([x_train, df_train.drop(columns_y, axis=1)], ignore_index=True)
        y_train = pd.concat([y_train, df_train.drop(columns_x, axis=1)], ignore_index=True)
        x_test = pd.concat([x_test, df_test.drop(columns_y, axis=1)], ignore_index=True)
        y_test = pd.concat([y_test, df_test.drop(columns_x, axis=1)], ignore_index=True)

    return x_train, y_train, x_test, y_test