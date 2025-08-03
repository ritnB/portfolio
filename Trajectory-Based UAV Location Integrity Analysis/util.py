import pandas as pd

# to normalize the values, identify a value to move datapoints near zero.
def move_to_zero(vector:pd.Series):
    vector=vector.copy(deep=True)
    desc=vector.describe()
    min=desc['min']
    max=desc['max']
    d=min+(max-min)/2
    vector-=d
    return vector

#Show center of the data points
def get_center_points(matrix:pd.DataFrame):
    center_points=[]
    for col in matrix:
        desc=matrix[col].describe()
        min=desc['min']
        max=desc['max']
        center_points.append(min+(max-min)/2)
    return center_points

#centralize every columns
def move_to_zero_all(matrix:pd.DataFrame, _target:list=[]):
    assert _target==[] or set([ _ in matrix.columns for _ in _target])=={True}
    # -> see here https://stackoverflow.com/questions/52583025/how-to-compare-each-element-of-two-lists-in-python
    matrix=matrix.copy(deep=True)
    for col in matrix:
        if col in _target:
            matrix[col]=move_to_zero(matrix[col])
    return matrix

def is_list_strings(list_str):
    if set([type(_) is str for _ in list_str])==set([True]):
        return True
    else:
        return False
def is_same_length(list_str, length=None):
    if length is None:
        criterion=len(set([len(_) for _ in list_str]))==1
    else:
        criterion=set([len(_)==length for _ in list_str])=={True}
    if criterion==True:
        return True
    else:
        return False

#getting parsed timestamp
def get_parsed_timestamp(df, inplace:bool=False):
    if inplace==False:
        df=df.copy(deep=True)
    stamp=df['timestamp']
    if not is_list_strings(stamp):
        print(-1)
        return -1
    if not is_same_length(stamp, length=7): #         mm:ss.s
        print(-2)
        return -2
    segment=[(_[0:2], _[3:5], _[6:7]) for _ in stamp]
    parsed_timestamp=[]
    for mm, ss, s in segment:
        time=int(mm)*60+int(ss)+round(int(s)*0.1, 3)
        parsed_timestamp.append(time)
    df['timestamp']=parsed_timestamp
    return df

#parsing, drop, regulization
def process_dataframe(df):
    # String timestamp to float timestamp
    df_parsed = get_parsed_timestamp(df, inplace=False)
    # Drop useless columns
    df_drop = df_parsed.drop(['time_utc_usec', 'fix_type', 'jamming_state', 'vel_ned_valid', 'timestamp_time_relative', 'heading', 'heading_offset', 'selected'], axis=1, inplace=False)
    # Move to center (regularization)
    df_regu = move_to_zero_all(df_drop, _target=['lat', 'lon', 'alt', 'alt_ellipsoid'])
    # Drop timestamp, utc
    # df_preprocessed = df_regu.drop(['timestamp'], axis=1, inplace=False)
    df_preprocessed = df_regu
    return df_preprocessed

# reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

# split dataset
def split_dataframe(df):
    diff = np.absolute(df['timestamp'].diff().values)
    diff[0] = 0.0
    diff = pd.Series(diff.round(decimals=5))
    split_indices = np.where(diff >= 1.0)[0]
    split_indices = [0] + list(split_indices) + [len(diff)]
    split_dfs = []

    for i, split_index in enumerate(split_indices):
        if i == len(split_indices) - 1:
            break
        start_index = split_index
        end_index = split_indices[i + 1]
        split_dfs.append(df.iloc[start_index:end_index])

    return split_dfs