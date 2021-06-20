
import torch
from torchvision import transforms
torch.manual_seed(1)
from ignite.contrib.handlers import LRScheduler, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OptimizerParamsHandler
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping
from pytorch_hebbian import config
from pytorch_hebbian.learning_rules import KrotovsRule
from pytorch_hebbian.optimizers import Local
from pytorch_hebbian.trainers import HebbianTrainer, SupervisedTrainer
from pytorch_hebbian.evaluators import HebbianEvaluator
import model_funcs
import load_without_faces
import cornet
import pdb

from torch.multiprocessing import set_start_method

#set_start_method('spawn', force=True)

from torch.utils.tensorboard import SummaryWriter

def save_model(model, epoch, loss, file_path):

    print('Saving model ...')
    #torch.save(model.state_dict(), f'{weights_dir}/cornet_classify_{cond}_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        }, file_path)

writer = SummaryWriter()
print_file = False   

#Image directory
model_type = 'hebbian'
cond = 'vggface'
writer = SummaryWriter(f'runs/{model_type}_{cond}')
image_dir = f"/scratch/vayzenbe/{cond}/"

image_dir = f"/lab_data/behrmannlab/image_sets/{cond}/"
weights_dir =f"/lab_data/behrmannlab/vlad/ginn/model_weights"

#Transformations for ImageNet
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

#These are all the default learning parameters from the run_CorNet script
lr = .01 #Starting learning rate
step_size = 10 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = 30
n_save = 5 #save model every X epochs

n_classes = 600 #Set number of classes the model will learn
start_epoch = 0 #set start_epoch (0 means, starting fresh)
model = model_funcs.load_model('hebbian',cond,start_epoch, weights_dir, n_classes) #load model


train_dir = image_dir + 'train'
val_dir = image_dir + 'val'
exclude_im = f"/lab_data/behrmannlab/image_sets/imagenet_face_files.csv"
exclude_folder = f"/lab_data/behrmannlab/image_sets/imagenet_animal_classes.csv"
train_dataset = load_without_faces.load_stim(train_dir, exclude_im, exclude_folder, transform=transform)
val_dataset = load_without_faces.load_stim(val_dir, exclude_im, exclude_folder, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)


print('data loaded')
learning_rule = KrotovsRule()
optimizer = Local(named_params=model.named_parameters(), lr=0.01)

evaluator = HebbianEvaluator(model=model, score_name='accuracy',
                                score_function=lambda engine: engine.state.metrics['accuracy'], epochs=1, supervised_from=-1)

trainer = HebbianTrainer(model=model, learning_rule=learning_rule, optimizer=optimizer, supervised_from=-1, device='cuda')

evaluator.attach(trainer.engine, Events.EPOCH_COMPLETED(every=1), trainloader, valloader)

trainer.run(train_loader=trainloader, epochs=1)


acc = evaluator.engine.state.metrics['accuracy']
loss = evaluator.engine.state.metrics['loss']

print(acc, loss)



epoch = 1
writer.add_scalar("Average Validation Loss", loss, epoch) #write to tensorboard
writer.add_scalar("Average Acc", acc, epoch) #write to tensorboard


file_path = f'{weights_dir}/cornet_{model_type}_{cond}_{epoch}.pt'
save_model(trainer.model, epoch,  loss, file_path)

for epoch in range(2, n_epochs+1):

        evaluator = HebbianEvaluator(model=trainer.model, score_name='accuracy',
                                score_function=lambda engine: engine.state.metrics['accuracy'], epochs=1, supervised_from=-1)

        trainer = HebbianTrainer(model=trainer.model, learning_rule=learning_rule, optimizer=optimizer, supervised_from=-1, device='cuda')

        evaluator.attach(trainer.engine, Events.EPOCH_COMPLETED(every=1), trainloader, valloader)

        trainer.run(train_loader=trainloader, epochs=1)

        acc = evaluator.engine.state.metrics['accuracy']
        loss = evaluator.engine.state.metrics['loss']

        #print(acc, loss)
        writer.add_scalar("Average Validation Loss", loss, epoch) #write to tensorboard
        writer.add_scalar("Average Acc", acc, epoch) #write to tensorboard

        if epoch % n_save == 0:
                file_path = f'{weights_dir}/cornet_{model_type}_{cond}_{epoch}.pt'
                save_model(trainer.model, epoch,  loss, file_path)
        


