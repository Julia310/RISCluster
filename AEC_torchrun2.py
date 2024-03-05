from MLTools.dist_training import Trainer, dist_train, prepare_dataloader


def load_train_objs():
    transform_pipeline = transforms.Compose([
        ZarrDataset.SpecgramNormalizer(transform='sample_norm_cent'),
        lambda x: x.double(),
    ])
    full_dataset = ZarrDataset('/work/users/jp348bcyy/rhoneDataCube/Cube_chunked_5758.zarr', 4, transform=transform_pipeline)  # load your dataset
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset into training and test sets
    train_set, test_set = random_split(full_dataset, [train_size, test_size])

    model = AEC()
    model.apply(init_weights)
    model = model.double()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return train_set, test_set, model, optimizer

def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    train_set, test_set, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(train_set, batch_size)
    test_data = prepare_dataloader(test_set, batch_size)

    trainer = Trainer(model, train_data, test_data, optimizer, save_every, snapshot_path)
    dist_train(trainer, total_epochs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=7, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size)