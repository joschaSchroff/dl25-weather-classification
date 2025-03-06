import train as train
import time
import json

def check_if_allready_done(args):
    try:
        with open('sweep.json') as f:
            pass
    except FileNotFoundError:
        with open('sweep.json', 'w') as f:
            json.dump([], f)

    with open('sweep.json') as f:
        data = json.load(f)
        for item in data:
            if item == args.wandb_name:
                return True
    return False

def add_to_json(args):
    with open('sweep.json') as f:
        data = json.load(f)
        data.append(args.wandb_name)
    with open('sweep.json', 'w') as f:
        json.dump(data, f)

def sweep(pause_time=600, pause_every=10):
    num_workers = [4]
    accelerator = ['cuda']
    seed = [42]
    num_epochs = [20]
    batch_size = [32]
    learning_rate = [0.01, 0.001, 0.0001]
    lr_step_size = [1]
    lr_gamma = [0.1,0.5,1.0]
    model_name = ['efficientnetb0', 'efficientnetb1', 'mobilenet', 'resnet50', 'vit']
    num_classes = [11]
    val_before_train = [False]
    early_stopping = [True]
    early_stopping_patience = [3]
    early_stopping_metric = ['val_loss']
    early_stopping_mode = ['min']
    early_stopping_delta = [0.001]
    data_dir = ['./data/weather-dataset']
    use_class_weights = [True,False]
    save_model = [False]
    save_model_path = ['./models']
    wandb_project = ['sweep-models']
    wandb_entity = ['dl25-weather']

    run_counter = 0
    for lr in learning_rate:
        for lr_g in lr_gamma:
            for model in model_name:
                for class_weights in use_class_weights:
                    wandb_name = [f"{model}-lr-{lr}-lr_g-{lr_g}-class_weights-{class_weights}-seed-{seed[0]}"]
                    args = train.TrainArgs(
                        num_workers=num_workers[0],
                        accelerator=accelerator[0],
                        seed=seed[0],
                        num_epochs=num_epochs[0],
                        batch_size=batch_size[0],
                        learning_rate=lr,
                        lr_step_size=lr_step_size[0],
                        lr_gamma=lr_g,
                        model_name=model,
                        num_classes=num_classes[0],
                        val_before_train=val_before_train[0],
                        early_stopping=early_stopping[0],
                        early_stopping_patience=early_stopping_patience[0],
                        early_stopping_metric=early_stopping_metric[0],
                        early_stopping_mode=early_stopping_mode[0],
                        early_stopping_delta=early_stopping_delta[0],
                        data_dir=data_dir[0],
                        use_class_weights=class_weights,
                        save_model=save_model[0],
                        save_model_path=save_model_path[0],
                        wandb_project=wandb_project[0],
                        wandb_name=wandb_name[0],
                        wandb_entity=wandb_entity[0]
                    )
                    if check_if_allready_done(args):
                        print(f"Already done - {args.wandb_name}")
                        continue
                    train.main(args)
                    add_to_json(args)
                    run_counter += 1
                    if run_counter % pause_every == 0:
                        print(f"Waiting for {pause_time} seconds")
                        time.sleep(pause_time) # sleep for x seconds for gpu to cool down


if __name__ == "__main__":
    sweep()