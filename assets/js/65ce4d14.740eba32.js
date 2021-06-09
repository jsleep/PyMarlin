(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[5627],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return p},kt:function(){return d}});var a=t(7294);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function r(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function s(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?r(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function o(e,n){if(null==e)return{};var t,a,i=function(e,n){if(null==e)return{};var t,a,i={},r=Object.keys(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||(i[t]=e[t]);return i}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var l=a.createContext({}),c=function(e){var n=a.useContext(l),t=n;return e&&(t="function"==typeof e?e(n):s(s({},n),e)),t},p=function(e){var n=c(e.components);return a.createElement(l.Provider,{value:n},e.children)},u={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},f=a.forwardRef((function(e,n){var t=e.components,i=e.mdxType,r=e.originalType,l=e.parentName,p=o(e,["components","mdxType","originalType","parentName"]),f=c(t),d=i,m=f["".concat(l,".").concat(d)]||f[d]||u[d]||r;return t?a.createElement(m,s(s({ref:n},p),{},{components:t})):a.createElement(m,s({ref:n},p))}));function d(e,n){var t=arguments,i=n&&n.mdxType;if("string"==typeof e||i){var r=t.length,s=new Array(r);s[0]=f;var o={};for(var l in n)hasOwnProperty.call(n,l)&&(o[l]=n[l]);o.originalType=e,o.mdxType="string"==typeof e?e:i,s[1]=o;for(var c=2;c<r;c++)s[c]=t[c];return a.createElement.apply(null,s)}return a.createElement.apply(null,t)}f.displayName="MDXCreateElement"},301:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return s},metadata:function(){return o},toc:function(){return l},default:function(){return p}});var a=t(2122),i=t(9756),r=(t(7294),t(3905)),s={sidebar_label:"implementation",title:"plugins.hf_seq_classification.implementation"},o={unversionedId:"reference/plugins/hf_seq_classification/implementation",id:"reference/plugins/hf_seq_classification/implementation",isDocsHomePage:!1,title:"plugins.hf_seq_classification.implementation",description:"HfSeqClassificationPlugin Objects",source:"@site/docs/reference/plugins/hf_seq_classification/implementation.md",sourceDirName:"reference/plugins/hf_seq_classification",slug:"/reference/plugins/hf_seq_classification/implementation",permalink:"/PyMarlin/docs/reference/plugins/hf_seq_classification/implementation",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/plugins/hf_seq_classification/implementation.md",version:"current",sidebar_label:"implementation",frontMatter:{sidebar_label:"implementation",title:"plugins.hf_seq_classification.implementation"},sidebar:"referenceSideBar",previous:{title:"plugins.hf_seq_classification.data_classes",permalink:"/PyMarlin/docs/reference/plugins/hf_seq_classification/data_classes"},next:{title:"plugins.hf_seq_classification.module_classes",permalink:"/PyMarlin/docs/reference/plugins/hf_seq_classification/module_classes"}},l=[{value:"HfSeqClassificationPlugin Objects",id:"hfseqclassificationplugin-objects",children:[]}],c={toc:l};function p(e){var n=e.components,t=(0,i.Z)(e,["components"]);return(0,r.kt)("wrapper",(0,a.Z)({},c,t,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h2",{id:"hfseqclassificationplugin-objects"},"HfSeqClassificationPlugin Objects"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"class HfSeqClassificationPlugin(Plugin)\n")),(0,r.kt)("p",null,"Plugin for Text Sequence Classification using Huggingface models."),(0,r.kt)("p",null,"plugin.setup() bootstraps the entire pipeline and returns a fully setup trainer."),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Example"),":"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"trainer = plugin.setup()\ntrainer.train()\ntrainer.validate()\n")),(0,r.kt)("p",null,"  Alternatively, you can run ",(0,r.kt)("inlineCode",{parentName:"p"},"setup_datainterface")," ",(0,r.kt)("inlineCode",{parentName:"p"},"setup_module")," ",(0,r.kt)("inlineCode",{parentName:"p"},"setup_trainer")," individually."),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"Example"),":"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"},"plugin.setup_datainterface()\nplugin.setup_module()\ntrainer = plugin.setup_trainer()\n")),(0,r.kt)("h4",{id:"__init__"},"_","_","init","_","_"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"}," | __init__(config: Optional[Dict] = None)\n")),(0,r.kt)("p",null,"CustomArgParser parses YAML config located at cmdline --config_path. If --config_path\nis not provided, assumes YAML file is named config.yaml and present in working directory.\nInstantiates dataclasses:\nself.data_args (arguments.DataInterfaceArguments): Instantiated dataclass containing\nargs required to initialize HfSeqClassificationDataInterface and HfSeqClassificationProcessor\nclasses.\nself.module_args (arguments.ModuleInterfaceArguments): Instantiated dataclass containing\nargs required to initialize HfSeqClassificationModule class.\nself.distill_args (arguments.DistillationArguments): Instantiated dataclass\nrequired to initialize DistillHfModule.\nSet self.distill_args.enable = True in config file to do knowledge distillation\ninstead of regular training.\nSets properties:\nself.datainterface: data_interface.DataInterface ","[HfSeqClassificationDataInterface]"," object\nself.dataprocessor: data_interface.DataProcessor ","[HfSeqClassificationProcessor]"," object.\nThese two together are used to read raw data and create sequences of tokens in ",(0,r.kt)("inlineCode",{parentName:"p"},"setup_datainterface"),".\nThe processed data is fed to HuggingFace AutoModelForSequenceClassification models.\nself.module: module_interface.ModuleInterface ","[HfSeqClassificationModule]"," object\nThis is used to initialize a Marlin trainer."),(0,r.kt)("h4",{id:"setup_datainterface"},"setup","_","datainterface"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"}," | setup_datainterface()\n")),(0,r.kt)("p",null,"Executes the data processing pipeline. Tokenizes train and val datasets using the\n",(0,r.kt)("inlineCode",{parentName:"p"},"dataprocessor")," and ",(0,r.kt)("inlineCode",{parentName:"p"},"datainterface"),".\nFinally calls ",(0,r.kt)("inlineCode",{parentName:"p"},"datainterface.setup_datasets(train_data, val_data)"),"."),(0,r.kt)("p",null,"Assumptions:\nTraining and validation files are placed in separate directories.\nAccepted file formats: json, tsv, csv.\nYAML config file should specify the column names or ids:\ndata:\ntext_a_col\ntext_b_col (optional None)\nlabel_col (optional None)\nHeader row is skipped for tsv/csv file if data_args.header = True\ndata_args.hf_tokenizer: String corresponding to Huggingface AutoTokenizer\ndata_args.cpu_threads: Number of processes to use for Python CPU multiprocessing"),(0,r.kt)("h4",{id:"setup_module"},"setup","_","module"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"}," | setup_module()\n")),(0,r.kt)("p",null,"Sets ",(0,r.kt)("inlineCode",{parentName:"p"},"HfSeqClassificationModule.data")," property to ",(0,r.kt)("inlineCode",{parentName:"p"},"datainterface")," which contains\nthe processed datasets. Assertion error is thrown if ",(0,r.kt)("inlineCode",{parentName:"p"},"datainterface")," retrieves no train\nor val data, indicating that ",(0,r.kt)("inlineCode",{parentName:"p"},"datainterface")," hasn","'","t been setup with processed data.\nSets the ",(0,r.kt)("inlineCode",{parentName:"p"},"HfSeqClassificationModule.model")," property after initializing weights:\nOption 1: Load weights from specified files mentioned in YAML config\nmodel:\nmodel_config_path\nmodel_config_file\nmodel_path\nmodel_file\nOption 2: Load from Huggingface model hub, specify string in YAML config as:\nmodel:\nhf_model\nIf distill_args.enable = True\nstudent: ",(0,r.kt)("inlineCode",{parentName:"p"},"HfSeqClassificationModule.model"),"\nteacher: ",(0,r.kt)("inlineCode",{parentName:"p"},"HfSeqClassificationModule.teacher"),"\nBoth student and teacher architectures must be Huggingface transformers."),(0,r.kt)("h4",{id:"setup"},"setup"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-python"}," | setup()\n")),(0,r.kt)("p",null,"Executes all the setup methods required to create a trn.Trainer object.\nTrainer needs ",(0,r.kt)("inlineCode",{parentName:"p"},"moduleinterface")," and backend is specified by self.trainer_args.backend."))}p.isMDXComponent=!0}}]);