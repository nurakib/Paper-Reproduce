import torch 

review = "I am at home, it feel grat....ahhh! SWETT hom,e----"
review = " ".join(review.split())

print(review)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)