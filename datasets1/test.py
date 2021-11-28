import pickle

with open("my_profile.txt", "wb") as myprofile:
    pickle.dump({"name": "AlwaysJane", "age": "20+", "sex": "female"}, myprofile)

with open("my_profile.txt", "rb") as get_myprofile:
    print(pickle.load(get_myprofile))