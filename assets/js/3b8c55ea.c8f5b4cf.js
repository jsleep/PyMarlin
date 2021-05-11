(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[217],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return u},kt:function(){return d}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var c=r.createContext({}),p=function(e){var t=r.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},u=function(e){var t=p(e.components);return r.createElement(c.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,l=e.originalType,c=e.parentName,u=o(e,["components","mdxType","originalType","parentName"]),m=p(n),d=i,f=m["".concat(c,".").concat(d)]||m[d]||s[d]||l;return n?r.createElement(f,a(a({ref:t},u),{},{components:n})):r.createElement(f,a({ref:t},u))}));function d(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var l=n.length,a=new Array(l);a[0]=m;var o={};for(var c in t)hasOwnProperty.call(t,c)&&(o[c]=t[c]);o.originalType=e,o.mdxType="string"==typeof e?e:i,a[1]=o;for(var p=2;p<l;p++)a[p]=n[p];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},872:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return a},metadata:function(){return o},toc:function(){return c},default:function(){return u}});var r=n(2122),i=n(9756),l=(n(7294),n(3905)),a={},o={unversionedId:"installation",id:"installation",isDocsHomePage:!1,title:"Installation",description:"In this guide, we will share instructions on how to set up pymarlin in the following environments:",source:"@site/docs/installation.md",sourceDirName:".",slug:"/installation",permalink:"/docs/installation",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/installation.md",version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"Getting Started",permalink:"/docs/getting-started"},next:{title:"PyMarlin in Pictures",permalink:"/docs/marlin-in-pictures"}},c=[{value:"Local/Dev Machine",id:"localdev-machine",children:[{value:"Environment setup",id:"environment-setup",children:[]},{value:"Install pytorch",id:"install-pytorch",children:[]},{value:"Install PyMarlin",id:"install-pymarlin",children:[]}]},{value:"AzureML",id:"azureml",children:[]}],p={toc:c};function u(e){var t=e.components,n=(0,i.Z)(e,["components"]);return(0,l.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("p",null,"In this guide, we will share instructions on how to set up pymarlin in the following environments:"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"Local/Dev Machine"),(0,l.kt)("li",{parentName:"ul"},"AzureML")),(0,l.kt)("h2",{id:"localdev-machine"},"Local/Dev Machine"),(0,l.kt)("h3",{id:"environment-setup"},"Environment setup"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"conda create -n pymarlin python=3.8\nconda activate pymarlin\n")),(0,l.kt)("h3",{id:"install-pytorch"},"Install pytorch"),(0,l.kt)("p",null,(0,l.kt)("a",{parentName:"p",href:"https://pytorch.org/get-started/locally/"},"Latest documentation")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"conda install pytorch cpuonly -c pytorch\n")),(0,l.kt)("h3",{id:"install-pymarlin"},"Install PyMarlin"),(0,l.kt)("p",null,"You can install from our internal pip or alternatively install from source."),(0,l.kt)("h4",{id:"install-from-pip"},"Install from pip"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"pip install pymarlin\n")),(0,l.kt)("h4",{id:"install-from-source"},"Install from source"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"git clone https://github.com/microsoft/PyMarlin.git\ncd PyMarlin\npip install -e .\n")),(0,l.kt)("h2",{id:"azureml"},"AzureML"),(0,l.kt)("p",null,"Specify the pip package in a supplied conda_env.yml file."))}u.isMDXComponent=!0}}]);