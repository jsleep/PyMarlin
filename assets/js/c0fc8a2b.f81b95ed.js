(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[937],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return c},kt:function(){return _}});var a=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function s(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?s(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):s(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},s=Object.keys(e);for(a=0;a<s.length;a++)n=s[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(a=0;a<s.length;a++)n=s[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var o=a.createContext({}),p=function(e){var t=a.useContext(o),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=p(e.components);return a.createElement(o.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},d=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,s=e.originalType,o=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),d=p(n),_=r,u=d["".concat(o,".").concat(_)]||d[_]||m[_]||s;return n?a.createElement(u,i(i({ref:t},c),{},{components:n})):a.createElement(u,i({ref:t},c))}));function _(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var s=n.length,i=new Array(s);i[0]=d;var l={};for(var o in t)hasOwnProperty.call(t,o)&&(l[o]=t[o]);l.originalType=e,l.mdxType="string"==typeof e?e:r,i[1]=l;for(var p=2;p<s;p++)i[p]=n[p];return a.createElement.apply(null,i)}return a.createElement.apply(null,n)}d.displayName="MDXCreateElement"},9095:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return i},metadata:function(){return l},toc:function(){return o},default:function(){return c}});var a=n(2122),r=n(9756),s=(n(7294),n(3905)),i={},l={unversionedId:"examples/cifar",id:"examples/cifar",isDocsHomePage:!1,title:"CIFAR image classification",description:"Krishan Subudhi 01/27/2021",source:"@site/docs/examples/cifar.md",sourceDirName:"examples",slug:"/examples/cifar",permalink:"/docs/examples/cifar",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/examples/cifar.md",version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"Bart Summarization",permalink:"/docs/examples/bart"},next:{title:"Data interface single and multi process",permalink:"/docs/examples/datamodule-example"}},o=[{value:"Step 1. Data preprocessing",id:"step-1-data-preprocessing",children:[]},{value:"Step 2. Training",id:"step-2-training",children:[{value:"Train for few steps",id:"train-for-few-steps",children:[]},{value:"Final Training",id:"final-training",children:[]}]},{value:"Step 3. Saving the model",id:"step-3-saving-the-model",children:[]}],p={toc:o};function c(e){var t=e.components,i=(0,r.Z)(e,["components"]);return(0,s.kt)("wrapper",(0,a.Z)({},p,i,{components:t,mdxType:"MDXLayout"}),(0,s.kt)("p",null,"Krishan Subudhi 01/27/2021"),(0,s.kt)("p",null,"This tutorial is based on official PyTorch blog on ",(0,s.kt)("a",{parentName:"p",href:"https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py"},"Training a classifier")," which trains a image classifier using CIFAR data."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"#!pip install torchvision\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport torch.optim as optim\nimport torchvision\nimport torchvision.transforms as transforms\nimport matplotlib.pyplot as plt\nimport numpy as np\n")),(0,s.kt)("h2",{id:"step-1-data-preprocessing"},"Step 1. Data preprocessing"),(0,s.kt)("p",null,"This step involves, "),(0,s.kt)("ol",null,(0,s.kt)("li",{parentName:"ol"},"Downloading data"),(0,s.kt)("li",{parentName:"ol"},"Preprocessing it "),(0,s.kt)("li",{parentName:"ol"},"Analyzing it"),(0,s.kt)("li",{parentName:"ol"},"Creating a final dataset")),(0,s.kt)("p",null,"In pymarlin, ",(0,s.kt)("strong",{parentName:"p"},"DataInteface")," and ",(0,s.kt)("strong",{parentName:"p"},"DataProcessor")," is where you implement the code related to all the steps above."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.core import data_interface\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class CifarDataProcessor(data_interface.DataProcessor):\n    def process(self):\n        transform = transforms.Compose(\n            [transforms.ToTensor(),\n             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])\n        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,\n                                        download=True, transform=transform)\n        testset = torchvision.datasets.CIFAR10(root='./data', train=False,\n                                       download=True, transform=transform)\n        return {'Train': trainset, 'Test': testset}\n\n    def analyze(self, datasets):\n        print(f'train data size = {len(datasets[\"Train\"])}')\n        print(f'val data size = {len(datasets[\"Test\"])}')\n        print('Examples')\n        sample_images = [datasets['Train'][i][0] for i in range(4)]\n        self._imshow(torchvision.utils.make_grid(sample_images))\n        \n    def _imshow(self,img):\n        img = img / 2 + 0.5     # unnormalize\n        npimg = img.numpy()\n        plt.imshow(np.transpose(npimg, (1, 2, 0)))\n        plt.show()\n\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class CifarDataInterface(data_interface.DataInterface):\n    \n    def setup_datasets(self, train_ds, val_ds):\n        self.train_ds = train_ds\n        self.val_ds = val_ds\n        \n    @property\n    def classes(self):\n        return ('plane', 'car', 'bird', 'cat',\n           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')\n    \n    def get_train_dataset(self):\n        return self.train_ds\n    \n    def get_val_dataset(self):\n        return self.val_ds \n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"dp = CifarDataProcessor()\ndm = CifarDataInterface()\ndatasets = dm.process_data(dp)\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"Files already downloaded and verified\nFiles already downloaded and verified\ntrain data size = 50000\nval data size = 10000\nExamples\n")),(0,s.kt)("p",null,(0,s.kt)("img",{alt:"png",src:n(363).Z})),(0,s.kt)("h2",{id:"step-2-training"},"Step 2. Training"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.core import module_interface\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class Net(nn.Module):\n    def __init__(self):\n        super(Net, self).__init__()\n        self.conv1 = nn.Conv2d(3, 6, 5)\n        self.pool = nn.MaxPool2d(2, 2)\n        self.conv2 = nn.Conv2d(6, 16, 5)\n        self.fc1 = nn.Linear(16 * 5 * 5, 120)\n        self.fc2 = nn.Linear(120, 84)\n        self.fc3 = nn.Linear(84, 10)\n\n    def forward(self, x):\n        x = self.pool(F.relu(self.conv1(x)))\n        x = self.pool(F.relu(self.conv2(x)))\n        x = x.view(-1, 16 * 5 * 5)\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"class CifarModule(module_interface.ModuleInterface):\n    '''\n    ModuleInterface contains instruction to create data loader , \n    defines train step, optimizer, scheduler, evaluation etc.\n    \n    Just implement the abstract function: refer docstrings.\n    '''\n    def __init__(self, data_interface):\n        super().__init__() # always initialize superclass first\n        self.data_interface = data_interface\n        \n        self.net = Net()\n        self.criterion = nn.CrossEntropyLoss()\n        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)\n        \n        self.running_loss = 0.0\n\n    def get_optimizers_schedulers(\n        self, estimated_global_steps_per_epoch: int, epochs: int\n        ):\n        return [self.optimizer], []\n\n    def get_train_dataloader(\n        self, sampler:type, batch_size:int\n        ):\n        print('Inside get_train_dataloader',batch_size)\n        return torch.utils.data.DataLoader(self.data_interface.get_train_dataset(), batch_size=batch_size,\n                                          shuffle=True)\n\n    def get_val_dataloaders(\n        self, sampler:torch.utils.data.Sampler, batch_size : int\n        ): \n        return torch.utils.data.DataLoader(self.data_interface.get_val_dataset(), batch_size=batch_size,\n                                         shuffle=False)\n\n    def train_step(\n        self, global_step: int, batch, device\n        ):\n        '''\n        First output should be loss. Can return multiple outputs\n        '''\n        inputs, labels = batch # output of dataloader will be input of train_step\n        inputs = inputs.to(device)\n        labels = labels.to(device)\n        outputs = self.net(inputs)\n        loss = self.criterion(outputs, labels)\n        self.running_loss += loss.item()\n        if global_step % 2000 == 0:    # print every 2000 mini-batches\n            print('[%5d] loss: %.3f' %\n                  (global_step, self.running_loss / 2000))\n            self.running_loss = 0.0\n        return loss\n\n    def val_step(self, global_step: int, batch, device) :\n        '''\n        Can return multiple outputs. First output need not be loss.\n        '''\n        images, labels = batch\n        images = images.to(device)\n        labels = labels.to(device)\n        outputs = self.net(images)\n        _, predicted = torch.max(outputs.data, 1)\n        total = labels.size(0)\n        correct = (predicted == labels).sum().item()\n        return correct, total\n\n    def on_end_val_epoch(self,\n        global_step: int,\n        *val_step_collated_outputs,\n        key='default'):\n        '''\n        callback after validation loop ends\n        '''\n        corrects, totals = val_step_collated_outputs\n        correct = sum(corrects) # list of integers\n        total= sum(totals)\n        \n        accuracy = 100 * correct / total\n        print(f'Val accuracy at step {global_step} = {accuracy}%')\n        \n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"dm.setup_datasets(datasets['Train'], datasets['Test'])\nmodule = CifarModule(dm)\n")),(0,s.kt)("h3",{id:"train-for-few-steps"},"Train for few steps"),(0,s.kt)("p",null,"Check if the entire loop runs without error. Use ",(0,s.kt)("inlineCode",{parentName:"p"},"max_train_steps_per_epoch")," and ",(0,s.kt)("inlineCode",{parentName:"p"},"max_val_steps_per_epoch")," to stop early. Set them to ",(0,s.kt)("inlineCode",{parentName:"p"},"null")," to train on full data."),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.core import trainer, trainer_backend\nfrom pymarlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments\nbackend = trainer_backend.SingleProcess()\nchkp_args = DefaultCheckpointerArguments(checkpoint=False)\n\nargs = trainer.TrainerArguments(\n    epochs=2,\n    max_train_steps_per_epoch = 100,\n    max_val_steps_per_epoch = 10,\n    train_batch_size=4,\n    val_batch_size=16,\n    writers=[],\n    log_level = 'DEBUG'\n)\ntr = trainer.Trainer(\n    trainer_backend = backend,\n    module = module,\n    args = args\n)\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"Inside get_train_dataloader 4\nInside get_train_dataloader 4\nSystemLog: 2021-04-01 21:39:57,419:INFO : pymarlin.core.trainer : 219 : _abc_impl: <_abc_data object at 0x000002489C5B1B70>\nSystemLog: 2021-04-01 21:39:57,420:INFO : pymarlin.core.trainer : 219 : args: TrainerArguments(epochs=2, use_gpu=True, train_batch_size=4, gpu_batch_size_limit=512, val_batch_size=16, max_train_steps_per_epoch=100, max_val_steps_per_epoch=10, clip_grads=True, max_grad_norm=1.0, reset_optimizers_schedulers=False, checkpointer_args=DefaultCheckpointerArguments(checkpoint=True, delete_existing_checkpoints=False, period=1, save_dir='your\\\\path', model_state_save_dir='your\\\\path', load_dir=None, load_filename=None, file_prefix='model', file_ext='pt', log_level='INFO'), distributed_training_args=DistributedTrainingArguments(local_rank=0, global_rank=0, world_size=1, backend='nccl', init_method='env://', gather_frequency=None), writers=[], stats_args=StatInitArguments(log_steps=1, update_system_stats=False, log_model_steps=1000, exclude_list='bias|LayerNorm|layer\\\\.[3-9]|layer\\\\.1(?!1)|layer\\\\.2(?!3)'), writer_args=WriterInitArguments(tb_log_dir='logs', tb_logpath_parent_env=None, tb_log_multi=False, tb_log_hist_steps=20000, model_log_level='INFO'), disable_tqdm=False, log_level='DEBUG', backend='sp')\nSystemLog: 2021-04-01 21:39:57,421:INFO : pymarlin.core.trainer : 219 : device: cpu\nSystemLog: 2021-04-01 21:39:57,421:INFO : pymarlin.core.trainer : 219 : estimated_global_steps_per_epoch: 100\nSystemLog: 2021-04-01 21:39:57,421:INFO : pymarlin.core.trainer : 219 : global_steps_finished: 0\nSystemLog: 2021-04-01 21:39:57,422:INFO : pymarlin.core.trainer : 219 : gradient_accumulation: 1\nSystemLog: 2021-04-01 21:39:57,422:INFO : pymarlin.core.trainer : 219 : is_distributed: False\nSystemLog: 2021-04-01 21:39:57,423:INFO : pymarlin.core.trainer : 219 : is_main_process: True\nSystemLog: 2021-04-01 21:39:57,424:INFO : pymarlin.core.trainer : 219 : logger: <Logger pymarlin.core.trainer (DEBUG)>\nSystemLog: 2021-04-01 21:39:57,424:INFO : pymarlin.core.trainer : 219 : pergpu_global_batch_size: 4\nSystemLog: 2021-04-01 21:39:57,424:INFO : pymarlin.core.trainer : 219 : stats: <pymarlin.utils.stats.basic_stats.BasicStats object at 0x000002489C594610>\nSystemLog: 2021-04-01 21:39:57,425:INFO : pymarlin.core.trainer : 219 : total_steps_finished: 0\nSystemLog: 2021-04-01 21:39:57,425:INFO : pymarlin.core.trainer : 219 : train_step_batch_size: 4\nSystemLog: 2021-04-01 21:39:57,425:INFO : pymarlin.core.trainer : 219 : trainer_backend: <pymarlin.core.trainer_backend.SingleProcess object at 0x000002489A69F3A0>\nSystemLog: 2021-04-01 21:39:57,426:INFO : pymarlin.core.trainer : 219 : val_step_batch_size: 16\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"tr.train()\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"  0%|          | 0/2 [00:00<?, ?it/s]\n\nSystemLog: 2021-04-01 21:40:03,871:INFO : pymarlin.core.trainer : 141 : Training epoch 0\n\n  0%|          | 0/12500 [00:00<?, ?batch/s]\n\nSystemLog: 2021-04-01 21:40:04,457:INFO : pymarlin.core.trainer : 147 : Validating  \n\n  0%|          | 0/625 [00:00<?, ?it/s]\n\nVal accuracy at step 100 = 15.625%\nSystemLog: 2021-04-01 21:40:04,515:INFO : pymarlin.core.trainer : 141 : Training epoch 1    \n\n  0%|          | 0/12500 [00:00<?, ?batch/s]\n\nSystemLog: 2021-04-01 21:40:05,096:INFO : pymarlin.core.trainer : 147 : Validating \n\n  0%|          | 0/625 [00:00<?, ?it/s]\n\nVal accuracy at step 200 = 10.0%\nSystemLog: 2021-04-01 21:40:05,160:INFO : pymarlin.core.trainer : 161 : Finished training ..\n")),(0,s.kt)("h3",{id:"final-training"},"Final Training"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"from pymarlin.core import trainer, trainer_backend\nfrom pymarlin.utils.checkpointer.checkpoint_utils import DefaultCheckpointerArguments\nbackend = trainer_backend.SingleProcess()\nchkp_args = DefaultCheckpointerArguments(checkpoint=False)\n\nargs = trainer.TrainerArguments(\n    epochs=2,\n    train_batch_size=4,\n    val_batch_size=16,\n    writers=['tensorboard'],\n    clip_grads=False,\n    log_level = 'INFO',\n    checkpointer_args=chkp_args\n)\n\ntr = trainer.Trainer(\n    trainer_backend = backend,\n    module = module,\n    args = args\n)\n\ntr.train()\n")),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre"},"Inside get_train_dataloader 4\nInside get_train_dataloader 4\nSystemLog: 2021-04-01 21:40:25,953:INFO : pymarlin.core.trainer : 219 : _abc_impl: <_abc_data object at 0x000002489C5B1B70>\nSystemLog: 2021-04-01 21:40:25,953:INFO : pymarlin.core.trainer : 219 : args: TrainerArguments(epochs=2, use_gpu=True, train_batch_size=4, gpu_batch_size_limit=512, val_batch_size=16, max_train_steps_per_epoch=None, max_val_steps_per_epoch=None, clip_grads=False, max_grad_norm=1.0, reset_optimizers_schedulers=False, checkpointer_args=DefaultCheckpointerArguments(checkpoint=False, delete_existing_checkpoints=False, period=1, save_dir='your\\\\path', model_state_save_dir='your\\\\path', load_dir=None, load_filename=None, file_prefix='model', \nfile_ext='pt', log_level='INFO'), distributed_training_args=DistributedTrainingArguments(local_rank=0, global_rank=0, world_size=1, backend='nccl', init_method='env://', gather_frequency=None), writers=['tensorboard'], stats_args=StatInitArguments(log_steps=1, update_system_stats=False, log_model_steps=1000, exclude_list='bias|LayerNorm|layer\\\\.[3-9]|layer\\\\.1(?!1)|layer\\\\.2(?!3)'), writer_args=WriterInitArguments(tb_log_dir='logs', tb_logpath_parent_env=None, tb_log_multi=False, tb_log_hist_steps=20000, model_log_level='INFO'), disable_tqdm=False, log_level='INFO', backend='sp')\nSystemLog: 2021-04-01 21:40:25,953:INFO : pymarlin.core.trainer : 219 : device: cpu\nSystemLog: 2021-04-01 21:40:25,954:INFO : pymarlin.core.trainer : 219 : estimated_global_steps_per_epoch: 12501\nSystemLog: 2021-04-01 21:40:25,954:INFO : pymarlin.core.trainer : 219 : global_steps_finished: 0\nSystemLog: 2021-04-01 21:40:25,954:INFO : pymarlin.core.trainer : 219 : gradient_accumulation: 1\nSystemLog: 2021-04-01 21:40:25,954:INFO : pymarlin.core.trainer : 219 : is_distributed: False\nSystemLog: 2021-04-01 21:40:25,955:INFO : pymarlin.core.trainer : 219 : is_main_process: True\nSystemLog: 2021-04-01 21:40:25,955:INFO : pymarlin.core.trainer : 219 : logger: <Logger pymarlin.core.trainer (INFO)>\nSystemLog: 2021-04-01 21:40:25,955:INFO : pymarlin.core.trainer : 219 : pergpu_global_batch_size: 4\nSystemLog: 2021-04-01 21:40:25,956:INFO : pymarlin.core.trainer : 219 : stats: <pymarlin.utils.stats.basic_stats.BasicStats object at 0x000002489C594610>\nSystemLog: 2021-04-01 21:40:25,957:INFO : pymarlin.core.trainer : 219 : total_steps_finished: 0\nSystemLog: 2021-04-01 21:40:25,957:INFO : pymarlin.core.trainer : 219 : train_step_batch_size: 4\nSystemLog: 2021-04-01 21:40:25,958:INFO : pymarlin.core.trainer : 219 : trainer_backend: <pymarlin.core.trainer_backend.SingleProcess object at 0x000002489CBD45E0>\nSystemLog: 2021-04-01 21:40:25,958:INFO : pymarlin.core.trainer : 219 : val_step_batch_size: 16\nInside get_train_dataloader 4\nSystemLog: 2021-04-01 21:40:25,960:INFO : pymarlin.utils.writer.tensorboard : 43 : Cleared directory logs (skipping azureml dirs)\nSystemLog: 2021-04-01 21:40:25,961:INFO : pymarlin.utils.writer.tensorboard : 46 : Created tensorboard folder logs : []\n\n  0%|          | 0/2 [00:00<?, ?it/s]\n\n  0%|          | 0/12500 [00:00<?, ?batch/s]\n\n[ 2000] loss: 2.395\n[ 4000] loss: 1.823\n[ 6000] loss: 1.653\n[ 8000] loss: 1.568\n[10000] loss: 1.495\n[12000] loss: 1.430\n\n  0%|          | 0/625 [00:00<?, ?it/s]\n\nVal accuracy at step 12500 = 49.7%\n\n  0%|          | 0/12500 [00:00<?, ?batch/s]\n\n[14000] loss: 1.406\n[16000] loss: 1.358\n[18000] loss: 1.324\n[20000] loss: 1.301\n[22000] loss: 1.288\n[24000] loss: 1.269\n\n  0%|          | 0/625 [00:00<?, ?it/s]\n\nVal accuracy at step 25000 = 55.35%\nSystemLog: 2021-04-01 21:43:59,683:INFO : pymarlin.core.trainer : 161 : Finished training ..\n")),(0,s.kt)("h2",{id:"step-3-saving-the-model"},"Step 3. Saving the model"),(0,s.kt)("pre",null,(0,s.kt)("code",{parentName:"pre",className:"language-python"},"PATH = './cifar_net.pth'\ntorch.save(module.net.state_dict(), PATH)\n")))}c.isMDXComponent=!0},363:function(e,t,n){"use strict";t.Z=n.p+"assets/images/cifar_1-a0316508726e8fa2f940588f4de5a8e8.png"}}]);