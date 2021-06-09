(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[7610],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return c},kt:function(){return d}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function s(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var o=r.createContext({}),u=function(e){var t=r.useContext(o),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},c=function(e){var t=u(e.components);return r.createElement(o.Provider,{value:t},e.children)},m={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},p=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,l=e.originalType,o=e.parentName,c=s(e,["components","mdxType","originalType","parentName"]),p=u(n),d=i,f=p["".concat(o,".").concat(d)]||p[d]||m[d]||l;return n?r.createElement(f,a(a({ref:t},c),{},{components:n})):r.createElement(f,a({ref:t},c))}));function d(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var l=n.length,a=new Array(l);a[0]=p;var s={};for(var o in t)hasOwnProperty.call(t,o)&&(s[o]=t[o]);s.originalType=e,s.mdxType="string"==typeof e?e:i,a[1]=s;for(var u=2;u<l;u++)a[u]=n[u];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}p.displayName="MDXCreateElement"},3705:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return a},metadata:function(){return s},toc:function(){return o},default:function(){return c}});var r=n(2122),i=n(9756),l=(n(7294),n(3905)),a={sidebar_label:"misc_utils",title:"utils.misc.misc_utils"},s={unversionedId:"reference/utils/misc/misc_utils",id:"reference/utils/misc/misc_utils",isDocsHomePage:!1,title:"utils.misc.misc_utils",description:"Miscellaneous utility functions",source:"@site/docs/reference/utils/misc/misc_utils.md",sourceDirName:"reference/utils/misc",slug:"/reference/utils/misc/misc_utils",permalink:"/PyMarlin/docs/reference/utils/misc/misc_utils",editUrl:"https://github.com/microsoft/PyMarlin/edit/master/website/docs/reference/utils/misc/misc_utils.md",version:"current",sidebar_label:"misc_utils",frontMatter:{sidebar_label:"misc_utils",title:"utils.misc.misc_utils"},sidebar:"referenceSideBar",previous:{title:"utils.logger.logging_utils",permalink:"/PyMarlin/docs/reference/utils/logger/logging_utils"},next:{title:"utils.stats.basic_stats",permalink:"/PyMarlin/docs/reference/utils/stats/basic_stats"}},o=[],u={toc:o};function c(e){var t=e.components,n=(0,i.Z)(e,["components"]);return(0,l.kt)("wrapper",(0,r.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("p",null,"Miscellaneous utility functions"),(0,l.kt)("h4",{id:"snake2camel"},"snake2camel"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"snake2camel(name)\n")),(0,l.kt)("p",null,"This method changes input name from snake format to camel format."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"name")," ",(0,l.kt)("em",{parentName:"li"},"str")," - snake format input name.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"name")," ",(0,l.kt)("em",{parentName:"li"},"str")," - camel format input name.")),(0,l.kt)("h4",{id:"clear_dir"},"clear","_","dir"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"clear_dir(path, skips=None)\n")),(0,l.kt)("p",null,"This method deletes the contents of the directory for which path\nhas been provided and not included in the skips list."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"path")," ",(0,l.kt)("em",{parentName:"li"},"str")," - Path for directory to be deleted."),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"skips")," ",(0,l.kt)("em",{parentName:"li"},"List","[str]")," - List of paths for sub directories to be skipped from deleting.")),(0,l.kt)("h4",{id:"debug"},"debug"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-python"},"debug(method)\n")),(0,l.kt)("p",null,"This method wraps input method with debug calls to measure time taken for\nthe given input method to finish."),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Arguments"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"method")," ",(0,l.kt)("em",{parentName:"li"},"function")," - Method which needs to be timed.")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"Returns"),":"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("inlineCode",{parentName:"li"},"debugged")," ",(0,l.kt)("em",{parentName:"li"},"method")," - debugged function.")))}c.isMDXComponent=!0}}]);