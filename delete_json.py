import os

for dirname in  os.listdir("./awe"):
    try:
        os.remove("./awe/" + dirname + "/annotations.json")
    except:
        print("Exception thrown. x does not exist.")
   