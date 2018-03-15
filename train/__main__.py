from dataset import HumanActionDataset
def main():
    d = HumanActionDataset("/dataset/human-action/images",(48,48,1),uses_generator=False)
    d.split_train_test()

if __name__ == '__main__':
    main()