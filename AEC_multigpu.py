import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from RISCluster.ZarrDataLoader import ZarrDataset
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import torch.distributed as dist
from RISCluster.networks import AEC, init_weights
import logging
# Suppress specific c10d warnings
import warnings
from tensorboardX import SummaryWriter

writer = SummaryWriter()

warnings.filterwarnings("ignore", category=UserWarning, module="c10d")

logging.getLogger('torch.distributed').setLevel(logging.ERROR)


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = F.mse_loss(output, batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)

        # Initialize tqdm only on rank 0
        pbar = tqdm(
            self.train_data,
            leave=True,
            desc="  Training",
            unit="batch",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in pbar:
            batch_size, mini_batch, channels, height, width = batch.size()
            batch = batch.view(batch_size * mini_batch, channels, height, width).to(self.gpu_id)
            loss = self._run_batch(batch)

            writer.add_scalar('training_loss', loss.item(), epoch)

        # Synchronize all processes before updating progress bar
        dist.barrier()

        # Close the tqdm progress bar for this epoch
        pbar.close()

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
        # Close the SummaryWriter when training is complete
        writer.close()


def load_train_objs():
    train_set = ZarrDataset('/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_5758.zarr', 4)  # load your dataset
    model = AEC()
    model.apply(init_weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)