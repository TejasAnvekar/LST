from collections import defaultdict
import utils
from Metrics import cluster_metrics as CM
from data import NPZ_Dataset
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings
from loss import loss as LOSS
from optim import optimizer as op
import argparse
import torch.nn.functional as F
from network import LST_AE, IDEC, LST_VAE
from sklearn import metrics
import sys
sys.path.insert(0, "/home/cvg-ws02/EXPERIMENTATION/Tejas/LST-master/")
warnings.simplefilter(action='ignore', category=FutureWarning)


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def cluster(model, loader, device):
    model.eval()
    z = []
    Y = []
    with torch.no_grad():
        for (x, y, _) in loader:
            x = x.to(device)
            _, Z, _ = model(x)
            z.extend(Z.detach().cpu().numpy())
            Y.extend(y.numpy())

    return np.array(z), np.array(Y)


def idec_plugin(model, batch_size, lr, optimizer, path, alpha, gamma, **kwargs):
    model = IDEC(model=model, n_z=10, n_clusters=kwargs["clusters"],
                 alpha=1, pretrain_path=path+"/Model/best_z1_best.pth.tar")
    model.pretrain()
    device = utils.set_seed_globally(
        seed_value=0, if_cuda=True, gpu=kwargs['gpu'])
    dataset = NPZ_Dataset(str=kwargs["dataset"])
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=kwargs["num_workers"])
    criterion_mse = LOSS(device, loss='bce')
    criterion_ssim = LOSS(device, loss='ssim', channel=3)
    optim = op(model, lr, optimizer)
    Optim = optim.call()

    z, y = cluster(model.ae, train_loader, device)

    z_metrics = CM.all_metrics(
        latent=z, y=y, n_clusters=kwargs["clusters"], n_init=20, n_jobs=-1)
    z_scores = z_metrics.scores()
    y_pred = z_metrics.y_pred
    print(f"ACC:{z_scores['acc']:.4f},NMI:{z_scores['nmi']:.4f}")
    z = None
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(
        z_metrics.kmeans.cluster_centers_).to(device)
    model.train()
    eval_dict2 = None
    results_idec = utils.save_results(path+"/IDEC/")

    MSE = []
    SSIM = []
    RECON = []
    DIVER = []

    ACC = []
    NMI = []
    ARI = []
    for epoch in range(kwargs["idec_epochs"]):
        mse = []
        ssim = []
        recon = []
        diver = []

        if epoch % kwargs["recenter"] == 0:
            tmp_q, y = cluster(model, train_loader, device)
            tmp_q = torch.tensor(tmp_q).to(device)
            # tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            y_pred = tmp_q.cpu().numpy().argmax(1)
            # y_pred = tmp_q.argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32)/y_pred.shape[0]
            y_pred_last = y_pred
            # y_pred = y_pred.cpu().numpy()

            y_pred = np.reshape(y_pred,(y_pred.shape[0]))
            y = np.reshape(y,(y.shape[0]))

            acc = CM.cluster_acc(y, y_pred)
            nmi = CM.nmi_score(y, y_pred)
            ari = CM.ari_score(y, y_pred)
            ACC.append(acc)
            NMI.append(nmi)
            ARI.append(ari)
            metric = {"acc": ACC, "nmi": NMI, "ari": ARI}
            if epoch != 0:
                results_idec.save_eval_metric(metric, "Z_idec")

            dict = {'weights': model.state_dict(),
                    'optimizer': Optim.state_dict(),
                    'epoch': epoch,
                    'nmi': nmi}

            results_idec.save_model(dict, f"LastEpoch")
            if (epoch) == 0:
                best_nmi = None

            best_nmi = save_best(model, nmi, results_idec,
                                 best_nmi, epoch, Optim, "best_z1", z, y)

            print(
                f"IDEC[{epoch}/{kwargs['idec_epochs']}] ==> ACC:{acc:.4f},NMI:{nmi:.4f},ARI:{ari:.4f}")
            if epoch > 0 and delta_label < kwargs["tol"]:
                print(f"delta_lable={delta_label},tol={kwargs['tol']}")
                print("reached tolerance")
                return
        for batch_idx, (x, _, idx) in enumerate(train_loader):
            x = x.to(device)
            idx = idx.to(device)
            x_bar, q, attention = model(x)
            mse_loss = criterion_mse(x_bar, x)
            ssim_loss = (1-criterion_ssim(x_bar, x))

            recon_loss = alpha * mse_loss + (1-alpha) * ssim_loss

            kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean')
            loss = gamma*kl_loss+recon_loss
            Optim.zero_grad()
            loss.backward()
            Optim.step()

            recon.append(recon_loss.item())
            mse.append(mse_loss.item())
            ssim.append(ssim_loss.item())
            diver.append(kl_loss.item())

        MSE.append(MEAN(mse))
        RECON.append(MEAN(recon))
        SSIM.append(MEAN(ssim))
        DIVER.append(MEAN(diver))

        loss_frame = {"MSE": MSE, "SSIM": SSIM, "DIVER": DIVER, "RECON": RECON}
        results_idec.save_loss(loss_frame)


def MEAN(x):
    return sum(x)/len(x)


def get_dict(glob, loc):
    if glob == None:
        glob = defaultdict(list)
        for key in loc.keys():
            glob[key] = [loc[key]]
    else:
        for key in loc.keys():
            glob[key].append(loc[key])

    return glob


def get_sample(path, device):
    f = np.load(path)
    x, y, = f['x_test'], f['y_test']

    return torch.tensor(x).to(device)


def save_best(model, nmi, results, best_nmi, epoch, optimizer, name, z, Y):
    if best_nmi == None:
        best_nmi = nmi
        dict = {'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'nmi': nmi}
        results.save_model(dict, name)
        results.save_latent("best", z, Y, name='Z')

        print("saved", best_nmi)

    elif nmi >= best_nmi:
        best_nmi = nmi
        dict = {'weights': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'nmi': nmi}

        results.save_model(dict, name)
        print("saved", best_nmi)

    return best_nmi


def main(**kwargs):

    for batch_size in kwargs["batch_sizes"]:
        for lr in kwargs["learning_rates"]:
            for optimizer in kwargs["optimizers"]:
                for alpha in kwargs["alphas"]:

                    eval_dict1 = None
                    r_path = kwargs["root"] + \
                        f"dataset:{kwargs['dataset']}_BS:{batch_size}_LR:{lr}_optim:{optimizer}_alpha:{alpha}_recentre:{kwargs['recenter']}"

                    results = utils.save_results(r_path)

                    device = utils.set_seed_globally(
                        seed_value=0, if_cuda=True, gpu=kwargs['gpu'])
                    dataset = NPZ_Dataset(str=kwargs["dataset"])

                    model = LST_VAE(
                        in_channels=3, image_size=96, patch_size=96//4, dim=64, num_classes=50, depth=1, token_dim=32, channel_dim=64, dec_layers=[200,1000,96*96*3], enc_layers=[20,10]
                    ).to(device)
                    print(model)

                    pretrain_loader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=kwargs["num_workers"])
                    criterion_mse = LOSS(device, loss='mse')
                    criterion_ssim = LOSS(device, loss='ssim', channel=3)
                    optim = op(model, lr, optimizer)
                    Optim = optim.call()

                    MSE = []
                    SSIM = []
                    RECON = []
                    VAE_LOSS = []

                    for epoch in range(kwargs["epochs"]):
                        loop = tqdm(enumerate(pretrain_loader), total=len(
                            pretrain_loader), leave=False, colour='green')
                        model.train()

                        mse = []
                        ssim = []
                        recon = []
                        vae_loss = []

                        for idx, (x, _, _) in loop:
                            x = x.to(device)
                            x_bar, z, attention = model(x)

                            Optim.zero_grad()

                            mse_loss = criterion_mse(x_bar, x)
                            ssim_loss = 1 - criterion_ssim(x_bar, x)
                            recon_loss = alpha*mse_loss + \
                                (1-alpha)*ssim_loss

                            vaeloss = model.vloss

                            loss = recon_loss + vaeloss
                            loss.backward()
                            Optim.step()

                            mse.append(mse_loss.item())
                            ssim.append(ssim_loss.item())
                            recon.append(recon_loss.item())
                            vae_loss.append(vaeloss.item())

                            if idx % kwargs["show"] == 0:
                                E = kwargs["epochs"]
                                loop.set_description(f"[{epoch}/{E}]")
                                loop.set_postfix(Recon_loss=recon_loss.item(
                                ), vae_loss=vaeloss.item())

                        # x_sample = get_sample(kwargs["sample_path"],device)
                        # with torch.no_grad():
                        #     model.eval()
                        #     x_hat,_,_ = model(x_sample)
                        #     results.save_images(epoch,x_sample,x_hat)
                        # model.train()

                        MSE.append(MEAN(mse))
                        SSIM.append(MEAN(ssim))
                        RECON.append(MEAN(recon))
                        VAE_LOSS.append(MEAN(vae_loss))

                        loss_frame = {"MSE": MSE, "SSIM": SSIM,
                                      "RECON": RECON, "VAE_LOSS": VAE_LOSS}
                        results.save_loss(loss_frame)

                        if (epoch+1) % kwargs["recenter"] == 0:
                            # X = torch.tensor(dataset.x)
                            Y = []
                            z, Y = cluster(model, pretrain_loader, device)

                            z_metrics = CM.all_metrics(
                                latent=z, y=Y, n_clusters=kwargs["clusters"], n_init=20, n_jobs=-1)

                            z_scores = z_metrics.scores()

                            eval_dict1 = get_dict(eval_dict1, z_scores)

                            results.save_eval_metric(eval_dict1, name='Z')

                            print(
                                f"z acc:{z_scores['acc']:.4f},nmi:{z_scores['nmi']:.4f}")

                            nmi = z_scores['nmi']

                            dict = {'weights': model.state_dict(),
                                    'optimizer': Optim.state_dict(),
                                    'epoch': epoch,
                                    'nmi': nmi}

                            results.save_model(dict, f"LastEpoch")
                            if (epoch+1) == kwargs["recenter"]:
                                best_nmi = None

                            best_nmi = save_best(
                                model, nmi, results, best_nmi, epoch, Optim, "best_z1", z, Y)

                            model.train()

    for batch_size in kwargs["batch_sizes"]:
        for lr in kwargs["learning_rates"]:
            for optimizer in kwargs["optimizers"]:
                for gamma in kwargs["gammas"]:
                    for alpha in kwargs["alphas"]:
                        idec_plugin(model, batch_size=batch_size, lr=lr,
                                    optimizer=optimizer, path=r_path, alpha=alpha, gamma=gamma, **kwargs)


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="LST")
    args.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="MSE*alpha + (1-aplha)*SSIM"

    )
    c = args.parse_args()
    main(
        root="/home/cvg-ws02/EXPERIMENTATION/Tejas/LST-master/STL10/logs/",
        # sample_path='/home/beast/DATA/DATASET_NPZ/FMNIST10.npz',
        epochs=200,
        idec_epochs=200,
        batch_sizes=[512,1024],
        learning_rates=[1e-3],
        optimizers=["adam"],
        alphas=[c.alpha],
        dataset="STL10",
        show=10,
        recenter=10,  # num epochs before dataset clustering
        clusters=10,
        gammas=[0.25,0.5,0.75,1],
        tol=0.0001,
        gpu=0,
        num_workers=20
    )
