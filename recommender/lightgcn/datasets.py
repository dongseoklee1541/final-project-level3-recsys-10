import os
import numpy as np
import pandas as pd
import torch

# DB로 부터 데이터를 불러오는 부분 | TODO DB 연결?
ratebeer = pd.read_json("ratebeer_korea.json")#.dropna()

######### data preprocessing #########
# 같은 유저가 두 번 이상 같은 아이템을 기록했다면 최종 점수를 통해서 결과를 기록한다.
data = ratebeer.sort_values(["profileName","reviewTime"]).drop_duplicates(subset=["beerName", "profileName"], keep="last")
data.index = range(len(data))

# 갖고 잇는 맥주의 데이터가 N개 이하인 유저는 학습에서 제외한다. 
N = 3
user_list = (data.groupby("profileName")["beerName"].nunique() > N).where(lambda x: x == True).dropna().index.tolist()
data = data[data["profileName"].isin(user_list)]
data.index = range(len(data))

# Scaling reviewScore(target)
data["reviewScore"] = data["reviewScore"]/5

# Remove NaN only for reviewScore(target)
data = data[~data["reviewScore"].isna()]
data.index = range(len(data))

# index 초기화
data.index = range(len(data))

######### 이 부분은 학습이 잘 되는지만 본다. ############
# # Get Candidate for [valid & test] |  >=10 and <= 15
# can_test = (data.groupby('profileName')["beerName"].count() >= 10 ) * (data.groupby('profileName')["beerName"].count() <= 15)
# can_test = list(can_test.where(lambda x: x == True).dropna().index)
# print("len(can_test) =",len(can_test))

# # Split users | 유저는 무작위로 2: 1: 1로 분할.
# np.random.shuffle(can_test) # 순서 섞기
# n_user_train = int(len(can_test)/2)
# n_user_valid = int(len(can_test)/4)
# n_user_test = len(can_test) - n_user_train - n_user_valid
# print(f"[user] train:valid:test = {n_user_train}:{n_user_valid}:{n_user_test}")

# # user_train = can_test[:n_user_train]
# user_valid = can_test[n_user_train:(n_user_valid+n_user_train)]
# user_test  = can_test[(n_user_valid+n_user_train):]


# # Get index for train valid, test | train은 다 사용하고, valid와 test 에서 각각 4개의 맥주 리뷰 데이터를 마킹한다.
# # (user_valid) 마지막에서 2개
# valid_index_last = data[data["profileName"].isin(user_valid)].sort_values(["profileName", "reviewTime"]).groupby("profileName").tail(2).index.tolist()
# # (user_valid) 마지막에서 2개 제외하고 무작위로 2개
# valid_index_rand = data[data["profileName"].isin(user_valid)].sort_values(["profileName", "reviewTime"]).groupby("profileName").head(-2).groupby("profileName").sample(2).index.tolist()
# valid_index = valid_index_last + valid_index_rand

# # (user_test) 마지막에서 2개
# test_index_last = data[data["profileName"].isin(user_test)].sort_values(["profileName", "reviewTime"]).groupby("profileName").tail(2).index.tolist()
# # (user_test) 마지막에서 2개 제외하고 무작위로 2개
# test_index_rand = data[data["profileName"].isin(user_test)].sort_values(["profileName", "reviewTime"]).groupby("profileName").head(-2).groupby("profileName").sample(2).index.tolist()
# test_index = test_index_last + test_index_rand


# # return Split Data
# train_data, valid_data, test_data = data.iloc[~data.index.isin(valid_index + test_index)], data.iloc[valid_index], data.iloc[test_index]

def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data["profileName"]))),
        sorted(list(set(data["beerName"]))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    # user_id와 겹치지 않게 item_id 만들기
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device):
    edge, label = [], []
    for user, item, acode in zip(data["profileName"], data["beerName"], data["reviewScore"]):
        uid, iid = id_2_index[user], id_2_index[item]
        # user와 item 사이의 edge
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    # label = torch.LongTensor(label)
    label = torch.FloatTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))


def process_data_inference(username, data, id_2_index, device):
    
    # username = data["profileName"][0] # 이 부분은 현재 프로그램 사용자가 될 것임
    itemname_list = data["beerName"].unique()
    iid_list = [id_2_index[item] for item in itemname_list]
    uid = id_2_index[username]

    inference_edge = [[uid, iid] for iid in iid_list]
    inference_edge = torch.LongTensor(inference_edge).T
    return inference_edge