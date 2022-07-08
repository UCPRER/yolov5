import json
obj = {
    'batch_size':34,
    'epochs':300,
    'img_size':640,
}
with open('hyp.json','w') as f:
    json.dump(obj=obj, fp=f)
