import os

# command = "pip install -r req.txt -i https://pypi.tuna.tsinghua.edu.cn/simple"
# result = os.system(command)
# result = os.system("pip install -U bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple")
# os.environ['HTTP_PROXY'] = 'http://127.0.0.21:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.21:7890'

from unimol_tools import MolTrain

trainer = MolTrain(
    task='multilabel_regression',
    data_type='molecule',
    model_name='unimolv2',
    model_size='164m',
    epochs=300,
    learning_rate=1e-4,
    batch_size=8,
    save_path='./multi_model',
    metrics="mse,mae,rmse,r2",
    kfold=10,
    split='scaffold',
    target_normalize='standard',
    smiles_col='smiles',
    target_cols='density,DetoD,DetoP,DetoQ,DetoT,DetoV,HOF_S,BDE',
    loss_key='uncertainty',
    use_wandb=False,
    wandb_project='unimol-mpp'
)

data_path = "train_set.csv"
trainer.fit(data_path)

print("训练完成！模型保存在 ./multi_model")
