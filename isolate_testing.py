import shutil
import os

lines = []

with open("awe-translation.csv") as file:
    lines = file.readlines()

for line in lines:
    split = line.split(",")
    c = int(split[0])-1
    filename = split[1]
    tt = split[2][0:3]

    print(c)
    print(filename)
    print(tt)

    if tt == "tes":
        f = "./awe/" + filename
        t = "./testing/" + str(c) + "_" + filename.split("/")[1]
        shutil.copyfile(f , t)
        os.remove(f)
