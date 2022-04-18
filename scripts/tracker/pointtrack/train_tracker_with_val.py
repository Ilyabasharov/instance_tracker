import os
import sys
import tqdm
import yaml
import click
import torch
import numpy as np
import random

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from datasets import get_dataset
from models.utils import get_embedding_model
from criterions.ranking import TrackingRankingLoss
from utils.utils import AverageMeter


def train_phase(
    dataloader,
    model,
    optimizer,
    loss_function,
    device,
) -> tuple:

    # define meters
    loss_meter = AverageMeter()
    loss_emb_meter = AverageMeter()

    # put model into training mode
    model.train()

    for sample in tqdm.tqdm(dataloader):

        points = sample['points'][0].to(device)
        xyxys = sample['xyxys'][0].to(device)
        labels = sample['labels'][0].to(device)

        outputs = model(points, xyxys)
        emb_loss = loss_function(outputs, labels)
        
        loss = emb_loss.mean()

        if loss.item() > 0:

            loss_emb_meter.update(emb_loss.mean().item())
            loss_meter.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss_meter.avg, loss_emb_meter.avg

@torch.no_grad()
def eval_phase(
    dataloader,
    model,
    loss_function,
    device,
) -> tuple:

    # define meters
    loss_meter = AverageMeter()
    loss_emb_meter = AverageMeter()

    # put model into training mode
    model.eval()

    for sample in tqdm.tqdm(dataloader):

        points = sample['points'][0].to(device)
        xyxys = sample['xyxys'][0].to(device)
        labels = sample['labels'][0].to(device)

        outputs = model(points, xyxys)
        emb_loss = loss_function(outputs, labels)

        loss = emb_loss.mean()

        if loss.item() > 0:

            loss_emb_meter.update(emb_loss.mean().item())
            loss_meter.update(loss.item())

    return loss_meter.avg, loss_emb_meter.avg

@click.command()
@click.option(
    '--config_path',
    default='config_mots/finetune_tracking.yaml',
    help='Path to the config.',
)
def main(
    config_path: str,
) -> None:

    with open(config_path, 'r') as file:
        args = yaml.load(
            stream=file,
            Loader=yaml.FullLoader,
        )

    # cudnn params
    torch.backends.cudnn.benchmark = True

    if args['cudnn']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # random seeds

    def seed_worker(
        worker_id: int,
    ) -> None:

        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()

    generator.manual_seed(args['seed'])
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    if torch.cuda.is_available():

        torch.cuda.manual_seed(args['seed'])
        torch.cuda.manual_seed_all(args['seed'])

    if args['save']:

        os.makedirs(
            name=args['save_dir'],
            exist_ok=True,
        )

    datasets = {
        phase: get_dataset(
            name=args[f'{phase}_dataset']['name'],
            dataset_opts=args[f'{phase}_dataset']['kwargs'],
        )
        for phase in ('train', 'val')
    }

    dataloaders = {
        phase: torch.utils.data.DataLoader(
            dataset=datasets[phase],
            batch_size=args[f'{phase}_dataset']['batch_size'],
            shuffle=True if phase == 'train' else False,
            num_workers=args[f'{phase}_dataset']['workers'],
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
        )
        for phase in ('train', 'val')
    }

    # define model
    model = torch.nn.DataParallel(get_embedding_model(
        params=args['model']['kwargs'],
        classname=args['class_name'],
    )).to(args['model']['kwargs']['device'])

    # set optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        **args['optimizer'],
    )
    
    #set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        **args['scheduler'],
    )

    #define loss function
    loss_function = TrackingRankingLoss(
        **args['loss_function'],
    )

    best_eval_loss = np.inf

    for epoch in range(args['start_epoch'], args['n_epochs']):

        train_loss, train_emb_loss = train_phase(dataloaders['train'], model, optimizer, loss_function, args['model']['kwargs']['device'])
        eval_loss, eval_emb_loss = eval_phase(dataloaders['val'], model, loss_function, args['model']['kwargs']['device'])

        info = f'Epoch {str(epoch).zfill(2)}/{str(args["n_epochs"]).zfill(2)} |'\
            f'Train Loss {train_loss:.7f} |'\
            f'Valid Loss {eval_loss:.7f} |'
        
        print(info)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if args['save']:
                torch.save(
                    obj={
                        'model_state_dict': model.state_dict(),
                    },
                    f=os.path.join(args['save_dir'], f'best_model_{args["class_name"]}.pth'),
                )
        
        scheduler.step()

if __name__ == '__main__':
    main()