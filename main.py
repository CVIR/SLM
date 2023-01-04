# Main script for Select, Label and Mix (SLM).
import params
from core import train_slm, warmstart_model, eval_model
from models import *
from utils import *
from datasets.office_31_PDA import *
from datasets.office_home_PDA import *
import argparse

parser = argparse.ArgumentParser(description='All arguments for the program.')

parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
parser.add_argument('--dataset_name', type=str, default='Office31', help='Name of the dataset from \'Office31\', \'OfficeHome\', \'VisDA2017\', \'ImageNetCaltech\' and \'CaltechImageNet\'.')
parser.add_argument('--src_dataset', type=str, default='amazon', help='Name of the SOURCE DOMAIN e.g. amazon.')
parser.add_argument('--tgt_dataset', type=str, default='webcam', help='Name of the TARGET DOMAIN e.g. webcam.')
parser.add_argument('--model_root', type=str, default='./checkpoints', help='Directory to save the models.')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for the framework.')
parser.add_argument('--learning_rate_ws', type=float, default=0.001, help='Learning rate for the Warmstart.')
parser.add_argument('--save_in_steps', type=int, default=50, help='Save models with this frequency.')
parser.add_argument('--log_in_steps', type=int, default=100, help='Log with this frequency.')
parser.add_argument('--eval_in_steps', type=int, default=1, help='Validate with this frequency.')
parser.add_argument('--momentum', type=float, default=0.9, help='For SGD optimizer.')
parser.add_argument('--model_name', type=str, default='resnet50', help='Name of the model.')
parser.add_argument('--classifier_name', type=str, default='resnet50', help='Name of the classifier.')
parser.add_argument('--source_images_path', type=str, default=None, help='Path to the list of source domain images. (ref: ./data_labels)')
parser.add_argument('--target_images_path', type=str, default=None, help='Path to the list of target domain images. (ref: ./data_labels)')
parser.add_argument('--num_iter_warmstart', type=int, default=5000, help='Number of iterations to warmstart.')
parser.add_argument('--num_iter_adapt', type=int, default=10000, help='Number of iterations to adapt (train SLM).')
parser.add_argument('--warmstart_models', type=str, default='True', help='Whether to warmstart the models or not.')
parser.add_argument('--pseudo_threshold', type=float, default=0.3, help='Threshold value for Label module.')
parser.add_argument('--manual_seed', type=int, default=None, help='Seed for Random Initialization.')

args = parser.parse_args()

if __name__ == '__main__':

    init_random_seed(args.manual_seed)

    params.batch_size = args.batch_size
    params.dataset_name = args.dataset_name
    params.src_dataset = args.src_dataset
    params.tgt_dataset = args.tgt_dataset
    params.model_root = args.model_root
    params.learning_rate = args.learning_rate
    params.learning_rate_ws = args.learning_rate_ws
    params.log_step_freq = args.log_in_steps
    params.log_step = args.log_in_steps
    params.eval_step_freq = args.eval_in_steps
    params.model_name = args.model_name
    params.classifier_name = args.classifier_name
    params.save_step_freq = args.save_in_steps
    params.momentum = args.momentum
    params.model_name = args.model_name
    params.classifier_name = args.classifier_name
    params.warmstart_models = args.warmstart_models
    params.pseudo_threshold = args.pseudo_threshold
    params.num_iter_warmstart = args.num_iter_warmstart
    params.num_iter_adapt = args.num_iter_adapt

    assert(params.model_name == params.classifier_name)

    if params.dataset_name=='Office31':
        src_data_loader, tgt_data_loader, tgt_data_loader_eval = get_office_31_PDA(source_path=args.source_images_path, target_path=args.target_images_path, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif params.dataset_name=='OfficeHome':
        src_data_loader, tgt_data_loader, tgt_data_loader_eval = get_office_home_PDA(source_path=args.source_images_path, target_path=args.target_images_path, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif params.dataset_name=='VisDA2017':
        src_data_loader, tgt_data_loader, tgt_data_loader_eval = get_visda17_PDA(source_path=args.source_images_path, target_path=args.target_images_path, batch_size=args.batch_size, shuffle=True, num_workers=2)
    elif params.dataset_name=='ImageNetCaltech' or params.dataset_name=='CaltechImageNet':
        src_data_loader, tgt_data_loader, tgt_data_loader_eval = get_imagenet_caltech_PDA(source_path=args.source_images_path, target_path=args.target_images_path, batch_size=args.batch_size, shuffle=True, num_workers=2)

    shared_model = get_shared_model(params.model_name, pretrained_val=True, dataset_name=params.dataset_name)
    shared_model.cuda()

    if params.warmstart_models=='True':
        print('=== Warmstarting Models ===')
        shared_model = warmstart_model(shared_model, src_data_loader, tgt_data_loader, tgt_data_loader_eval, num_iterations=params.num_iter_warmstart)
        print('Models warmstarted successfully...')

    print("=== Training SLM ===")
    print(">>> Model {} <<<".format(params.model_name))
    print(shared_model)

    shared_model = train_slm(shared_model, src_data_loader, tgt_data_loader, tgt_data_loader_eval, num_iterations=params.num_iter_adapt)
    
    print("=== Evaluating the Shared Model on Target Domain ===")
    temp_loss, temp_acc = eval_model(shared_model, tgt_data_loader_eval)

    print()
    print('Select, Label, Mix: SLM training for {}-->{} completed ... '.format(params.src_dataset, params.tgt_dataset))
    print()