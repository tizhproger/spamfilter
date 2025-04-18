from spam_detector import Detector

messages = ["You got 300$! Click on link below to redeem it.", "You've won a free prize!"]

clf = Detector("distilbert", models_dir="models")
result = clf.predict(messages, return_proba=True)

print(result)
