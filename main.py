import numpy as np
import torch
import model
import test
from data_pre import load_dataset, Grammar, zeropad_to_max_len
from sampling import get_coordinates_labels, get_train_test
from dataloader import simple_data_generator
import warnings
warnings.filterwarnings('ignore')
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
selection_rules = ["rect 7"]
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='n_epochs')
    parser.add_argument("--max_depth", type=int, default=2, help="max_depth")
    parser.add_argument('--batch_size', type=int, default=128, help="num_batches")
    parser.add_argument("--num_head", type=int, default=10)
    parser.add_argument("--drop_rate", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--log_every_n_samples", type=int, default=5)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="models")
    parser.add_argument("--test_size",type=float, default=0.99)
    parser.add_argument("--val_size", type=float, default=0.01)
    parser.add_argument("--start_learning_rate", type=float, default=3e-4)
    parser.add_argument("--dataset", type=str, default="FM")
    parser.add_argument("--prembed", type=bool, default=True)
    parser.add_argument("--prembed_dim", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="dataset/RIVER")
    parser.add_argument("--repeat_term", type=int, default=5)
    parser.add_argument("--is_valid", type=bool, default=False)
    parser.add_argument("--limited_num", type=int, default=50)
    parser.add_argument("--num_hidden", type=int, default=100)
    parser.add_argument("--masking", type=bool, default=False)
    parser.add_argument("--pooling", type=bool, default=False)
    parser.add_argument("--pool_size", type=int, default=3)
    parser.add_argument("--data_augment", type = bool, default=False)
    parser.add_argument("--max_len", type=int,default=49)
    parser.add_argument("--test_region", type=str, default="rect 7")
    parser.add_argument("--SLIC_scale", type=int, default=50)
    args = parser.parse_args()
    return args

def GT_To_One_Hot(gt):
    GT_One_Hot = []
    for i in range(gt.shape[0]):
        if gt[i] != 0 :
            temp = [0, 1]
        else:
            temp = [1, 0]
        GT_One_Hot.append(temp)
    return GT_One_Hot

global Dataset

Dataset = "RV"  # RV, FM, USA

sampling_mode = "random"
margin = 4

X1, X2, y, dataset_name, X3 = load_dataset(Dataset)
xshape = X1.shape[1:]
arg = get_args()
height, width, bands = X1.shape
OA_record = []
kappa_record = []
alltime_record = []

for repterm in range(arg.repeat_term):

    print("start")
    coords, labels = get_coordinates_labels(y)

    train_coords, train_labels, test_coords, test_labels, val_coords, val_labels = get_train_test(data=coords, data_labels=labels, val_size=arg.val_size,
                                                                          test_size=arg.test_size)
    train_coords = train_coords + margin
    test_coords = test_coords + margin
    val_coords = val_coords +margin

    X_train1, X_train2, X_train3 = Grammar(X1, X2, train_coords, X3, method="rect 7")
    X_val1, X_val2, X_val3 = Grammar(X1, X2, val_coords, X3, method = "rect 7")

    y_train = train_labels
    y_test = test_labels
    y_val = val_labels
    X_train_shape = X_train1.shape
    X_val_shape = X_val1.shape

    if len(X_train_shape) == 4:
        X_train1 = np.reshape(X_train1, [X_train_shape[0], X_train_shape[1] * X_train_shape[2],
                                       X_train_shape[3]])
        X_train2 = np.reshape(X_train2, [X_train_shape[0], X_train_shape[1] * X_train_shape[2],
                                         X_train_shape[3]])
        X_train3 = np.reshape(X_train3, [X_train_shape[0], X_train_shape[1] * X_train_shape[2],
                                         X_train_shape[3]])


        X_val1 = np.reshape(X_val1, [X_val_shape[0], X_val_shape[1] * X_val_shape[2], X_val_shape[3]])
        X_val2 = np.reshape(X_val2, [X_val_shape[0], X_val_shape[1] * X_val_shape[2], X_val_shape[3]])
        X_val3 = np.reshape(X_val3, [X_val_shape[0], X_val_shape[1] * X_val_shape[2], X_val_shape[3]])

    X_train1 = zeropad_to_max_len(X_train1, max_len=arg.max_len)
    X_train2 = zeropad_to_max_len(X_train2, max_len=arg.max_len)
    X_train3 = zeropad_to_max_len(X_train3, max_len=arg.max_len)
    X_val1 = zeropad_to_max_len(X_val1, max_len=arg.max_len)
    X_val2 = zeropad_to_max_len(X_val2, max_len=arg.max_len)
    X_val3 = zeropad_to_max_len(X_val3, max_len=arg.max_len)

    for i in range(2):
        print("num train and test in class %d is %d / %d" % (
        i, (y_train == i).sum(), (y_test == i).sum()))

    net = model.TriTF(arg.max_len, xshape[-1], arg.max_depth, arg.num_head, arg.num_hidden, arg.drop_rate, arg.attention_dropout,
                         arg.prembed, arg.prembed_dim, arg.masking, arg.pooling, arg.pool_size, arg.batch_size)

    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=arg.start_learning_rate)
    best_loss = 99999

    loss_fn = torch.nn.CrossEntropyLoss(reduce=False)

    for i in range(arg.n_epochs+1):
        net.train()
        train_generator = simple_data_generator(X_train1, X_train2, X_train3, y_train, batch_size=arg.batch_size, shuffle=True)
        val_generator = simple_data_generator(X_val1, X_val2, X_val3, y_val, batch_size=arg.batch_size, shuffle=True)
        sample_train_num = np.shape(y_train)
        sample_test_num = np.shape(y_test)
        sample_val_num = np.shape(y_val)
        total_train_eq = 0
        total_train = 0
        total_train_loss = 0
        train_total_batch = 0
        val_total_batch = 0
        total_val_eq = 0
        total_val_loss = 0
        for ind, (X_batch1, X_batch2, X_batch3, y_batch, adj) in enumerate(train_generator):
            X_batch1 = torch.from_numpy(X_batch1.astype(np.float32)).to(device)
            X_batch2 = torch.from_numpy(X_batch2.astype(np.float32)).to(device)
            X_batch3 = torch.from_numpy(X_batch3.astype(np.float32)).to(device)
            adj = torch.from_numpy(adj.astype(np.float32)).to(device)
            train_batch_num = X_batch1.shape[0]
            train_batch_num = train_batch_num
            train_total_batch = train_total_batch + train_batch_num
            y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(device)
            optimizer.zero_grad()
            output = net(X_batch1, X_batch2, X_batch3, adj)
            loss = loss_fn(output, y_batch.long())
            loss = torch.sum(loss)
            loss.backward(retain_graph=True)
            optimizer.step()
            loss_cpu = loss.cpu().detach().numpy()
            total_train_loss = loss_cpu + total_train_loss
            logits = torch.argmax(output, dim=1)
            equal_num = torch.eq(logits, y_batch)
            equal_num = torch.sum(equal_num)
            equal_num = equal_num.cpu().numpy()
            total_train_eq = total_train_eq + equal_num

        net.eval()
        for ind, (X_batch1, X_batch2, X_batch3, y_batch, adj) in enumerate(val_generator):
            with torch.no_grad():
                X_batch1 = torch.from_numpy(X_batch1.astype(np.float32)).to(device)
                X_batch2 = torch.from_numpy(X_batch2.astype(np.float32)).to(device)
                X_batch3 = torch.from_numpy(X_batch3.astype(np.float32)).to(device)
                adj = torch.from_numpy(adj.astype(np.float32)).to(device)
                val_batch_num = X_batch1.shape[0]
                val_batch_num = val_batch_num
                val_total_batch = val_total_batch + val_batch_num
                y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(device)
                output = net(X_batch1, X_batch2, X_batch3, adj)
                loss = loss_fn(output, y_batch.long())
                loss = torch.sum(loss)
                loss_cpu = loss.cpu().detach().numpy()
                total_val_loss = loss_cpu + total_val_loss
                logits = torch.argmax(output, dim=1)
                equal_num = torch.eq(logits, y_batch)
                equal_num = torch.sum(equal_num)
                equal_num = equal_num.cpu().numpy()
                total_val_eq = total_val_eq + equal_num

        train_oa = total_train_eq / train_total_batch
        val_oa = total_val_eq / val_total_batch
        print(i, "    train loss: ", total_train_loss, "    train oa: ", train_oa, "val loss", total_val_loss, "    val oa: ", val_oa)
        if total_val_loss < best_loss:
            best_loss = total_val_loss
            torch.save(net.state_dict(), "model\\best_model.pt")
            print('save model...')
            torch.cuda.empty_cache()

    output, test_OA, test_kappa = test.test(X1, X2, X3, test_coords, y_test, net, arg, y)
    print("test oa: ", test_OA, "      test kappa: ", test_kappa)



