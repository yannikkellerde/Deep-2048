with open("hyperparameter_tuning.log","r") as f:
    comb_list = [x.split(":") for x in f.read().splitlines()]
blub = {}
for c in comb_list:
    if c[0] in blub:
        blub[c[0]].append(int(c[1]))
    else:
        blub[c[0]]=[int(c[1])]
blub = {key:sum(value)/len(value) for key,value in blub.items()}
print(blub)