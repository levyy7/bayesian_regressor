from ucimlrepo import fetch_ucirepo

if __name__ == '__main__':
    # fetch dataset: Communities and Crime
    dataset = fetch_ucirepo(id=183)

    # data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    # metadata
    print(dataset.metadata)

    # variable information
    print(dataset.variables)
