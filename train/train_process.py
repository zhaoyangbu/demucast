import datetime
import time
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.cuda.amp import autocast, GradScaler

from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator

from sklearn import metrics


class DataTracker:
    def __init__(self):
        self.current_value = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.current_value = 0
        self.sum = 0
        self.count = 0

    def update(self, current_value, n=1):
        self.current_value = current_value
        self.sum += n * current_value
        self.count += n

    def average(self):
        return self.sum / self.count



def train(model: nn.Module, train_loader, test_loader, args):
    # gpu  training
    if not torch.cuda.is_available():
        raise Exception('Please check the cuda if it is available ')

    device = torch.device("cuda")

    # set the grad enable True
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = DataTracker()
    prepare_data_time = DataTracker()
    epoch_time = DataTracker()

    train_loss_tracker = DataTracker()
    learning_rate_tracker = DataTracker()
    val_acc_tracker = DataTracker()

    writer = SummaryWriter("/data/z_projects/awfe_v2/log/demucast")

    # step and epoch init
    global_step, start_epoch = 0, 1

    # load the model
    model = model.to(device)
    print('load the model to {}'.format(str(device)))

    # statistics the parameters
    print('statistics the parameters')
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    print('Total parameters number is {:.3f}'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    print('Total trainable parameters is {:.3f}'.format(sum(p.numel() for p in trainable_parameters) / 1e6))

    # Set up the optimizer
    weight_decay = 0
    betas = (0.95, 0.999)
    optimizer = torch.optim.Adam(trainable_parameters, args.lr, weight_decay=weight_decay, betas=betas)
    #optimizer = torch.optim.SGD(trainable_parameters, args.lr, weight_decay=weight_decay)
    print('Set the optimizer with learning rate {}; weight decay {}; betas {}'.format(args.lr, weight_decay, betas))

    # learning rate scheduler for dataset
    print('Training the Model on {}'.format(args.dataset))
    step_epoch = [12, 24, 50, 65]
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, step_epoch, gamma, last_epoch=-1)
    print('The Scheduler Setting: step epoch:{}, gamma{}'.format(step_epoch, gamma))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=5)



    # loss and metrics setting
    main_metrics = 'acc'
    citation = nn.CrossEntropyLoss()
    warmup = True
    print('training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(
        str(args.dataset), str(main_metrics), str(citation), str(scheduler)))

    # automatic mixed precision
    scaler = GradScaler()

    # start epoch training

    print("start training...")

    for epoch in range(start_epoch, args.num_epochs + 1):

        # print the current time
        print('-' * 15)
        print(datetime.datetime.now())
        print("current #steps=%s, #epochs=%s" % (global_step, epoch))
        model.train()

        train_p_bar = tqdm(enumerate(BackgroundGenerator(train_loader)),
                           total=len(train_loader))

        epoch_start_time = time.time()
        batch_start_time = time.time()
        for i, (audio_input, labels) in train_p_bar:
            # prepare the data

            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            prepare_data_time.update(time.time() - batch_start_time)

            # warmup for learning rate:
            if warmup and global_step <= args.warmup_steps and global_step % 50 == 0:
                warm_lr = (global_step / args.warmup_steps) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                # print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # AMP forward
            with autocast():
                output = model(audio_input)
                loss = citation(output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_tracker.update(loss.item())
            learning_rate_tracker.update(optimizer.param_groups[0]['lr'])
            writer.add_scalar('train/loss', train_loss_tracker.current_value, global_step)
            writer.add_scalar('train/lr', learning_rate_tracker.current_value, global_step)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            train_p_bar.set_description("Compute efficiency: {:.2f}, epoch: {}/{} lr: {} loss:{}".format(
                batch_time.current_value - prepare_data_time.current_value / batch_time.current_value, epoch,
                args.num_epochs, learning_rate_tracker.current_value, train_loss_tracker.current_value))

            global_step += 1

        print('start validation')
        test_p_bar = tqdm(enumerate(BackgroundGenerator(test_loader)),
                          total=len(test_loader))
        model.eval()
        predictions_list = []
        targets_list = []
        loss_list = []
        with torch.no_grad():
            for i, (audio_input, labels) in test_p_bar:
                audio_input = audio_input.to(device)

                # compute output
                audio_output = model(audio_input)
                prob_dist = torch.softmax(audio_output,dim=1) # prob distribution
                predictions = torch.argmax(prob_dist, dim=1).to('cpu').detach()

                predictions_list.append(predictions)
                targets_list.append(labels)

                labels = labels.to(device)
                loss = citation(audio_output, labels)
                loss_list.append(loss.to('cpu').detach())

            output = torch.cat(predictions_list)
            target = torch.cat(targets_list)
            print(len(output))
            print(len(target))
            # loss = np.mean(loss_list)

            # Accuracy
            acc = metrics.accuracy_score(target, output)

            val_acc_tracker.update(acc)
            #scheduler.step(acc)
            if main_metrics == 'acc':
                print("acc: {:.6f}".format(val_acc_tracker.current_value))
            writer.add_scalar('val/acc', val_acc_tracker.current_value, epoch)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), "/data/z_projects/awfe_v2/ckpt/demucast_model/audio_model{:d}.pth".format(epoch))
        scheduler.step()
        epoch_time.update(time.time() - epoch_start_time)
        epoch_start_time = time.time()

    print('average_epoch_time:{}'.format(epoch_time.average()))
